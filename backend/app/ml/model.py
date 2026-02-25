"""
Modèle de prédiction hippique.
- Ensemble LightGBM + XGBoost + CatBoost (classification binaire)
- LGBMRanker (LambdaRank) pour l'ordre
- Calibration Platt/Isotonique sur partition séparée (pas de leakage)
- Optuna pour le tuning des hyperparamètres
"""
import logging
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Course, Partant, Prediction, Cheval
from app.ml.features import build_features_for_course, FEATURE_NAMES

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "saved"
MODEL_PATH = MODEL_DIR / "model.pkl"

# Default hyperparams (overridden by Optuna)
_DEFAULT_LGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=42,
    verbose=-1,
)

_DEFAULT_XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=1.0,  # will be computed dynamically
    random_state=42,
    eval_metric="logloss",
    verbosity=0,
)

_DEFAULT_CAT_PARAMS = dict(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=0,
    auto_class_weights="Balanced",
)


class HippiquePredictor:
    def __init__(self):
        self.model = None          # ensemble dict or legacy model
        self.ranker = None         # LGBMRanker
        self.calibrator = None     # IsotonicRegression
        self.ensemble_weights = None
        self.best_params = None    # from Optuna
        self._load_model()

    def _load_model(self):
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "ensemble" in data:
                self.model = data["ensemble"]
                self.ranker = data.get("ranker")
                self.calibrator = data.get("calibrator")
                self.ensemble_weights = data.get("weights", [1.0])
                self.best_params = data.get("best_params")
                logger.info("Modèle ensemble chargé depuis %s", MODEL_PATH)
            else:
                # Legacy model (CalibratedClassifierCV)
                self.model = data
                logger.info("Modèle legacy chargé depuis %s", MODEL_PATH)

    def _save_model(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "ensemble": self.model,
            "ranker": self.ranker,
            "calibrator": self.calibrator,
            "weights": self.ensemble_weights,
            "best_params": self.best_params,
        }
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(data, f)
        logger.info("Modèle sauvegardé dans %s", MODEL_PATH)

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _build_dataset(courses, all_rows):
        """Flatten course rows into X, y, groups arrays."""
        X_list, y_list, groups = [], [], []
        for course, rows in zip(courses, all_rows):
            valid = [r for r in rows if r["is_winner"] is not None]
            if not valid:
                continue
            for r in valid:
                X_list.append(r["features"])
                y_list.append(1 if r["is_winner"] else 0)
            groups.append(len(valid))
        return np.array(X_list), np.array(y_list), groups

    @staticmethod
    def _build_ranking_labels(all_rows):
        """Build relevance labels for LambdaRank: higher = better."""
        labels = []
        for rows in all_rows:
            valid = [r for r in rows if r["is_winner"] is not None]
            if not valid:
                continue
            nb = len(valid)
            for r in valid:
                cl = r.get("classement")
                if cl is not None and cl > 0:
                    labels.append(max(nb - cl + 1, 0))
                elif r["is_winner"]:
                    labels.append(nb)
                else:
                    labels.append(0)
        return np.array(labels)

    @staticmethod
    def _split_by_groups(groups, train_frac=0.70, calib_frac=0.15):
        """Split course groups into train/calib/test indices."""
        total = sum(groups)
        cum = np.cumsum(groups)
        train_end = int(total * train_frac)
        calib_end = int(total * (train_frac + calib_frac))

        train_idx = 0
        for i, c in enumerate(cum):
            if c >= train_end:
                train_idx = i + 1
                break

        calib_idx = train_idx
        for i in range(train_idx, len(cum)):
            if cum[i] >= calib_end:
                calib_idx = i + 1
                break

        n_train = int(cum[train_idx - 1]) if train_idx > 0 else 0
        n_calib = int(cum[calib_idx - 1]) - n_train if calib_idx > train_idx else 0

        return n_train, n_calib, train_idx, calib_idx

    # ── Train ──────────────────────────────────────────────────────

    async def train(self, session: AsyncSession) -> dict:
        """Entraîne l'ensemble de modèles. Retourne les métriques."""
        if not HAS_LGB:
            return {"error": "LightGBM non disponible"}

        # Fetch all completed courses
        stmt = select(Course).where(Course.statut == "TERMINE").order_by(Course.date.asc())
        result = await session.execute(stmt)
        courses = result.scalars().all()

        if len(courses) < 20:
            return {"error": f"Pas assez de courses ({len(courses)}). Minimum 20."}

        # Build features for all courses
        all_rows = []
        for course in courses:
            rows = await build_features_for_course(session, course)
            all_rows.append(rows)

        X, y, groups = self._build_dataset(courses, all_rows)
        logger.info("Dataset : %d partants, %d gagnants (%.1f%%)", len(y), y.sum(), y.mean() * 100)

        # Split: train 70% / calib 15% / test 15%
        n_train, n_calib, g_train, g_calib = self._split_by_groups(groups)
        n_total = len(y)

        X_train, y_train = X[:n_train], y[:n_train]
        X_calib, y_calib = X[n_train:n_train + n_calib], y[n_train:n_train + n_calib]
        X_test, y_test = X[n_train + n_calib:], y[n_train + n_calib:]

        groups_train = groups[:g_train]
        groups_test = groups[g_calib:]

        if len(X_train) == 0 or len(X_calib) == 0 or len(X_test) == 0:
            return {"error": "Pas assez de données pour le split train/calib/test"}

        # ── Train classifiers ─────────────────────────────
        lgb_params = dict(_DEFAULT_LGB_PARAMS)
        if self.best_params and "lgb" in self.best_params:
            lgb_params.update(self.best_params["lgb"])

        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train)

        models = [lgb_model]
        model_names = ["LightGBM"]

        if HAS_XGB:
            xgb_params = dict(_DEFAULT_XGB_PARAMS)
            pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
            xgb_params["scale_pos_weight"] = pos_weight
            if self.best_params and "xgb" in self.best_params:
                xgb_params.update(self.best_params["xgb"])
            xgb_model = xgb.XGBClassifier(**xgb_params)
            xgb_model.fit(X_train, y_train)
            models.append(xgb_model)
            model_names.append("XGBoost")

        if HAS_CAT:
            cat_params = dict(_DEFAULT_CAT_PARAMS)
            if self.best_params and "cat" in self.best_params:
                cat_params.update(self.best_params["cat"])
            cat_model = CatBoostClassifier(**cat_params)
            cat_model.fit(X_train, y_train)
            models.append(cat_model)
            model_names.append("CatBoost")

        # ── Find optimal ensemble weights on calib set ────
        best_weights = self._find_ensemble_weights(models, X_calib, y_calib)
        self.ensemble_weights = best_weights
        logger.info("Poids ensemble : %s (%s)", best_weights, model_names)

        # ── Calibration isotonique on calib set ───────────
        blend_calib = self._blend_predict(models, best_weights, X_calib)
        self.calibrator = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
        self.calibrator.fit(blend_calib, y_calib)

        # ── LambdaRank ranker ─────────────────────────────
        self.ranker = None
        if len(groups_train) >= 5:
            try:
                rank_labels = self._build_ranking_labels(all_rows[:g_train])
                if len(rank_labels) == len(X_train):
                    ranker = lgb.LGBMRanker(
                        objective="lambdarank",
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=6,
                        num_leaves=31,
                        min_child_samples=10,
                        verbose=-1,
                    )
                    ranker.fit(X_train, rank_labels, group=groups_train)
                    self.ranker = ranker
                    logger.info("LGBMRanker entraîné avec succès")
            except Exception as e:
                logger.warning("Échec LGBMRanker : %s", e)

        self.model = models
        self._save_model()

        # ── Metrics on test set ───────────────────────────
        metrics = self._compute_metrics(
            models, best_weights, X_test, y_test, groups_test,
            total_partants=len(y),
        )

        # Feature importance (from LightGBM)
        if hasattr(lgb_model, "feature_importances_"):
            importances = lgb_model.feature_importances_
            fi = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
            metrics["feature_importance"] = {name: float(imp) for name, imp in fi}

        metrics["feature_names"] = FEATURE_NAMES
        metrics["ensemble_models"] = model_names
        metrics["ensemble_weights"] = [float(w) for w in best_weights]

        logger.info("Entraînement terminé : top1=%.1f%%, top3=%.1f%%, log_loss=%.4f",
                     metrics["top1_accuracy"], metrics["top3_accuracy"], metrics["log_loss"])
        return metrics

    def _blend_predict(self, models, weights, X):
        """Weighted average of model predict_proba."""
        blend = np.zeros(len(X))
        for m, w in zip(models, weights):
            blend += w * m.predict_proba(X)[:, 1]
        return blend / sum(weights)

    def _find_ensemble_weights(self, models, X, y):
        """Grid search for optimal blend weights (minimize log_loss)."""
        if len(models) == 1:
            return [1.0]

        probas = [m.predict_proba(X)[:, 1] for m in models]
        best_loss = float("inf")
        best_weights = [1.0] * len(models)

        # Grid search with step 0.1
        n = len(models)
        if n == 2:
            for w1 in range(0, 11):
                w = [w1 / 10, (10 - w1) / 10]
                blend = sum(p * wi for p, wi in zip(probas, w))
                blend = np.clip(blend, 1e-7, 1 - 1e-7)
                ll = log_loss(y, blend)
                if ll < best_loss:
                    best_loss = ll
                    best_weights = w
        elif n == 3:
            for w1 in range(0, 11):
                for w2 in range(0, 11 - w1):
                    w3 = 10 - w1 - w2
                    w = [w1 / 10, w2 / 10, w3 / 10]
                    blend = sum(p * wi for p, wi in zip(probas, w))
                    blend = np.clip(blend, 1e-7, 1 - 1e-7)
                    ll = log_loss(y, blend)
                    if ll < best_loss:
                        best_loss = ll
                        best_weights = w

        return best_weights

    def _compute_metrics(self, models, weights, X_test, y_test, groups_test, total_partants=0):
        """Compute enriched metrics on test set."""
        # Blend predictions
        blend = self._blend_predict(models, weights, X_test)
        calibrated = self.calibrator.transform(blend) if self.calibrator else blend

        # Scalar metrics
        ll = log_loss(y_test, np.clip(calibrated, 1e-7, 1 - 1e-7))
        brier = brier_score_loss(y_test, calibrated)

        # Per-course metrics
        top1_correct = 0
        top3_correct = 0
        total_races = 0
        mrr_sum = 0.0
        roi_flat_gains = 0.0
        roi_flat_mises = 0.0
        roi_value_gains = 0.0
        roi_value_mises = 0.0
        roi_kelly_gains = 0.0
        roi_kelly_mises = 0.0
        bankroll = 1000.0  # bankroll simulé pour Kelly

        # Récupérer les cotes originales depuis X_test (index 0 = cote_probable)
        cotes_test = X_test[:, 0] if X_test.shape[1] > 0 else np.full(len(X_test), np.nan)

        idx = 0
        for g in groups_test:
            if g == 0:
                continue
            course_probas = calibrated[idx:idx + g]
            course_labels = y_test[idx:idx + g]
            course_cotes = cotes_test[idx:idx + g]
            idx += g
            total_races += 1

            rankings = np.argsort(-course_probas)

            # Top-1
            if course_labels[rankings[0]] == 1:
                top1_correct += 1

            # Top-3
            if any(course_labels[rankings[i]] == 1 for i in range(min(3, len(rankings)))):
                top3_correct += 1

            # MRR
            for rank, ri in enumerate(rankings, 1):
                if course_labels[ri] == 1:
                    mrr_sum += 1.0 / rank
                    break

            # Favori du modèle
            fav_idx = rankings[0]
            fav_prob = course_probas[fav_idx]
            fav_cote = course_cotes[fav_idx]
            fav_won = course_labels[fav_idx] == 1

            # ROI mise fixe 1€ sur le favori
            roi_flat_mises += 1.0
            if fav_won and not np.isnan(fav_cote) and fav_cote >= 1.0:
                roi_flat_gains += fav_cote
            # else: gains += 0 (perte de la mise)

            # ROI Value Bet (ne mise que si prob > 1.3 × prob_implicite)
            if not np.isnan(fav_cote) and fav_cote >= 1.0:
                prob_impl = 1.0 / fav_cote
                if fav_prob > prob_impl * 1.3:
                    roi_value_mises += 1.0
                    if fav_won:
                        roi_value_gains += fav_cote

            # ROI Kelly (demi-Kelly, cap 10%)
            if not np.isnan(fav_cote) and fav_cote > 1.0:
                edge = fav_prob * fav_cote - 1.0
                if edge > 0:
                    kelly_f = min((edge / (fav_cote - 1.0)) * 0.5, 0.10)
                    mise_kelly = bankroll * kelly_f
                    roi_kelly_mises += mise_kelly
                    if fav_won:
                        roi_kelly_gains += mise_kelly * fav_cote

        metrics = {
            "total_partants": total_partants,
            "total_courses_test": total_races,
            "top1_accuracy": round(top1_correct / max(total_races, 1) * 100, 1),
            "top3_accuracy": round(top3_correct / max(total_races, 1) * 100, 1),
            "log_loss": round(ll, 4),
            "brier_score": round(brier, 4),
            "mrr": round(mrr_sum / max(total_races, 1), 4),
            "roi_flat": round((roi_flat_gains - roi_flat_mises) / max(roi_flat_mises, 1) * 100, 1),
            "roi_value_bet": round((roi_value_gains - roi_value_mises) / max(roi_value_mises, 1) * 100, 1) if roi_value_mises > 0 else None,
            "roi_value_bet_nb_mises": int(roi_value_mises),
            "roi_kelly": round((roi_kelly_gains - roi_kelly_mises) / max(roi_kelly_mises, 1) * 100, 1) if roi_kelly_mises > 0 else None,
        }
        return metrics

    # ── Tune (Optuna) ──────────────────────────────────────────────

    async def tune(self, session: AsyncSession, n_trials: int = 50) -> dict:
        """Hyperparameter tuning with Optuna. Returns best params."""
        if not HAS_OPTUNA:
            return {"error": "Optuna non installé"}
        if not HAS_LGB:
            return {"error": "LightGBM non disponible"}

        from sklearn.model_selection import TimeSeriesSplit

        # Build dataset
        stmt = select(Course).where(Course.statut == "TERMINE").order_by(Course.date.asc())
        result = await session.execute(stmt)
        courses = result.scalars().all()

        if len(courses) < 30:
            return {"error": f"Pas assez de courses ({len(courses)}). Minimum 30 pour le tuning."}

        all_rows = []
        for course in courses:
            rows = await build_features_for_course(session, course)
            all_rows.append(rows)

        X, y, groups = self._build_dataset(courses, all_rows)

        # Use only 80% for tuning (keep 20% untouched for final eval)
        n_tune = int(len(X) * 0.8)
        X_tune, y_tune = X[:n_tune], y[:n_tune]

        tscv = TimeSeriesSplit(n_splits=3)

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

            scores = []
            for train_idx, val_idx in tscv.split(X_tune):
                model = lgb.LGBMClassifier(
                    class_weight="balanced",
                    random_state=42,
                    verbose=-1,
                    **params,
                )
                model.fit(X_tune[train_idx], y_tune[train_idx])
                pred = model.predict_proba(X_tune[val_idx])[:, 1]
                pred = np.clip(pred, 1e-7, 1 - 1e-7)
                scores.append(log_loss(y_tune[val_idx], pred))
            return np.mean(scores)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best = study.best_params
        self.best_params = {"lgb": best}

        # Save best params with model
        if self.model:
            self._save_model()

        return {
            "status": "ok",
            "best_params": best,
            "best_log_loss": round(study.best_value, 4),
            "n_trials": n_trials,
        }

    # ── Predict ────────────────────────────────────────────────────

    async def predict_course(self, session: AsyncSession, course: Course) -> list[dict]:
        """Prédit les probabilités de victoire pour chaque partant."""
        if self.model is None:
            logger.warning("Aucun modèle entraîné, utilisation de la baseline (cote)")
            return await self._baseline_predict(session, course)

        rows = await build_features_for_course(session, course)
        if not rows:
            return []

        X = np.array([r["features"] for r in rows])

        # Get probabilities from ensemble
        if isinstance(self.model, list):
            blend = self._blend_predict(self.model, self.ensemble_weights or [1.0], X)
            if self.calibrator:
                probas = self.calibrator.transform(blend)
            else:
                probas = blend
        else:
            # Legacy model
            probas = self.model.predict_proba(X)[:, 1]

        # Get rankings from ranker if available
        rank_scores = None
        if self.ranker:
            try:
                rank_scores = self.ranker.predict(X)
            except Exception:
                rank_scores = None

        results = []
        for i, row in enumerate(rows):
            prob = float(probas[i])
            cote = row["features"][0]  # cote_probable
            # Guard NaN for value bet & Kelly
            if np.isnan(cote) or np.isnan(prob) or cote < 1.0:
                is_value = False
                kelly_fraction = 0.0
                kelly_mise = 0.0
            else:
                prob_implicite = 1.0 / max(cote, 1.0)
                # Value bet : prob modèle > 1.3× prob implicite (marge 30%)
                is_value = prob > prob_implicite * 1.3
                # Kelly Criterion : f = (p*b - 1) / (b - 1) où b = cote, p = prob modèle
                b = cote
                edge = prob * b - 1.0
                if edge > 0 and b > 1.0:
                    kelly_fraction = edge / (b - 1.0)
                    # Demi-Kelly pour limiter la variance
                    kelly_fraction = min(kelly_fraction * 0.5, 0.10)  # cap à 10% du bankroll
                    kelly_mise = round(kelly_fraction * 100, 2)  # en % du bankroll
                else:
                    kelly_fraction = 0.0
                    kelly_mise = 0.0

            results.append({
                "partant_id": row["partant_id"],
                "cheval_id": row["cheval_id"],
                "numero": row["numero"],
                "probabilite": prob,
                "is_value_bet": is_value,
                "kelly_fraction": kelly_fraction,
                "kelly_mise_pct": kelly_mise,
                "_rank_score": float(rank_scores[i]) if rank_scores is not None else prob,
            })

        # Sort by ranker score (uses LambdaRank if available)
        results.sort(key=lambda x: -x["_rank_score"])

        total_prob = sum(r["probabilite"] for r in results if not np.isnan(r["probabilite"]))
        for rank, r in enumerate(results, 1):
            r["rang_predit"] = rank
            del r["_rank_score"]
            if rank == 1 and len(results) > 1:
                ecart = r["probabilite"] - results[1]["probabilite"]
                r["score_confiance"] = min(round(ecart / max(r["probabilite"], 0.01) * 100, 0), 100)
            else:
                r["score_confiance"] = max(0, round(r["probabilite"] / max(total_prob, 0.01) * 100, 0))

        return results

    async def _baseline_predict(self, session: AsyncSession, course: Course) -> list[dict]:
        """Prédiction baseline : inversement proportionnelle à la cote."""
        stmt = select(Partant).where(
            Partant.course_id == course.id,
            Partant.statut == "PARTANT",
        )
        result = await session.execute(stmt)
        partants = result.scalars().all()

        results = []
        total_inv = sum(1.0 / max(p.cote_probable or 20, 1) for p in partants)

        for p in partants:
            cote = p.cote_probable or 20.0
            prob = (1.0 / max(cote, 1)) / max(total_inv, 0.01)
            results.append({
                "partant_id": p.id,
                "cheval_id": p.cheval_id,
                "numero": p.numero,
                "probabilite": prob,
                "is_value_bet": False,
            })

        results.sort(key=lambda x: -x["probabilite"])
        for rank, r in enumerate(results, 1):
            r["rang_predit"] = rank
            if rank == 1 and len(results) > 1:
                ecart = r["probabilite"] - results[1]["probabilite"]
                r["score_confiance"] = min(round(ecart / max(r["probabilite"], 0.01) * 100, 0), 100)
            else:
                r["score_confiance"] = round(r["probabilite"] * 100, 0)

        return results

    async def generate_comment(self, session: AsyncSession, partant_id: int, course: Course, rank: int) -> str:
        """Génère un commentaire justificatif pour une prédiction."""
        stmt = select(Partant).where(Partant.id == partant_id)
        result = await session.execute(stmt)
        partant = result.scalar_one_or_none()
        if not partant:
            return ""

        cheval_stmt = select(Cheval).where(Cheval.id == partant.cheval_id)
        cheval = (await session.execute(cheval_stmt)).scalar_one_or_none()
        cheval_nom = cheval.nom if cheval else "Inconnu"

        hist_stmt = (
            select(Partant)
            .join(Course)
            .where(
                Partant.cheval_id == partant.cheval_id,
                Course.statut == "TERMINE",
                Course.date < course.date,
            )
            .order_by(Course.date.desc())
            .limit(5)
        )
        hist = (await session.execute(hist_stmt)).scalars().all()

        nb_courses = len(hist)
        nb_victoires = sum(1 for h in hist if h.classement == 1)
        nb_places = sum(1 for h in hist if h.classement and 1 <= h.classement <= 3)

        parts = []
        if rank == 1:
            parts.append(f"{cheval_nom} est notre favori pour cette course.")
        else:
            parts.append(f"{cheval_nom} est classé #{rank} dans nos prévisions.")

        if nb_courses > 0:
            parts.append(f"Sur ses {nb_courses} dernières courses : {nb_victoires} victoire(s), {nb_places} place(s).")
        else:
            parts.append("Pas d'historique récent disponible.")

        cote = partant.cote_probable
        if cote:
            if cote <= 3:
                parts.append(f"Favori du public (cote {cote:.1f}).")
            elif cote <= 8:
                parts.append(f"Cote raisonnable ({cote:.1f}), bon rapport risque/gain.")
            else:
                parts.append(f"Outsider (cote {cote:.1f}), potentiel de gain élevé si victoire.")

        if course.distance:
            parts.append(f"Distance : {course.distance}m.")

        return " ".join(parts)

    async def predict_and_save(self, session: AsyncSession, course: Course):
        """Prédit et sauvegarde les prédictions pour une course."""
        predictions = await self.predict_course(session, course)

        for pred in predictions[:5]:  # Top 5
            comment = await self.generate_comment(
                session, pred["partant_id"], course, pred["rang_predit"]
            )
            db_pred = Prediction(
                course_id=course.id,
                partant_id=pred["partant_id"],
                probabilite=pred["probabilite"],
                rang_predit=pred["rang_predit"],
                score_confiance=pred["score_confiance"],
                is_value_bet=pred["is_value_bet"],
                commentaire=comment,
            )
            session.add(db_pred)

        await session.flush()

    async def backtest_course(self, session: AsyncSession, course: Course):
        """Met à jour les prédictions d'une course terminée avec les résultats réels."""
        stmt = select(Prediction).where(Prediction.course_id == course.id)
        result = await session.execute(stmt)
        preds = result.scalars().all()

        mise = 1.0

        for pred in preds:
            partant_stmt = select(Partant).where(Partant.id == pred.partant_id)
            partant = (await session.execute(partant_stmt)).scalar_one_or_none()
            if not partant:
                continue

            classement = partant.classement
            pred.resultat_gagnant = classement == 1
            pred.resultat_place = classement is not None and 1 <= classement <= 3

            if pred.rang_predit == 1:
                if pred.resultat_gagnant and partant.rapport_gagnant:
                    pred.gain_gagnant = partant.rapport_gagnant - mise
                else:
                    pred.gain_gagnant = -mise

                if pred.resultat_place and partant.rapport_place:
                    pred.gain_place = partant.rapport_place - mise
                else:
                    pred.gain_place = -mise

        await session.flush()
