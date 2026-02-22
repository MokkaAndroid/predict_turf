"""
Modèle de prédiction hippique.
Utilise LightGBM en mode classification binaire (gagnant / non-gagnant).
Walk-forward validation pour éviter le data leakage.
"""
import logging
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Course, Partant, Prediction, Cheval
from app.ml.features import build_features_for_course, FEATURE_NAMES

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "saved"
MODEL_PATH = MODEL_DIR / "model.pkl"


class HippiquePredictor:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            logger.info("Modèle chargé depuis %s", MODEL_PATH)

    def _save_model(self):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Modèle sauvegardé dans %s", MODEL_PATH)

    async def train(self, session: AsyncSession) -> dict:
        """
        Entraîne le modèle sur toutes les courses terminées.
        Retourne les métriques de validation.
        """
        if not HAS_LGB:
            logger.error("LightGBM non installé")
            return {"error": "LightGBM non disponible"}

        # Récupérer toutes les courses terminées, ordonnées par date
        stmt = (
            select(Course)
            .where(Course.statut == "TERMINE")
            .order_by(Course.date.asc())
        )
        result = await session.execute(stmt)
        courses = result.scalars().all()

        if len(courses) < 20:
            return {"error": f"Pas assez de courses ({len(courses)}). Minimum 20."}

        all_X = []
        all_y = []

        for course in courses:
            rows = await build_features_for_course(session, course)
            for row in rows:
                if row["is_winner"] is not None:
                    all_X.append(row["features"])
                    all_y.append(1 if row["is_winner"] else 0)

        X = np.array(all_X)
        y = np.array(all_y)

        logger.info("Dataset : %d partants, %d gagnants (%.1f%%)", len(y), y.sum(), y.mean() * 100)

        # Entraînement LightGBM
        base_model = lgb.LGBMClassifier(
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

        # Walk-forward validation (dernier 20% comme test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        base_model.fit(X_train, y_train)

        # Calibration Platt
        self.model = CalibratedClassifierCV(base_model, cv="prefit", method="sigmoid")
        self.model.fit(X_test, y_test)

        # Métriques
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        top1_correct = 0
        top3_correct = 0
        total_races = 0

        # Regrouper par course pour calculer top-1 et top-3
        idx = 0
        for course in courses[split_idx:]:
            rows = await build_features_for_course(session, course)
            valid_rows = [r for r in rows if r["is_winner"] is not None]
            n = len(valid_rows)
            if n == 0:
                continue

            course_probas = y_pred_proba[idx:idx + n]
            course_labels = y_test[idx:idx + n]
            idx += n

            if len(course_probas) != n:
                continue

            rankings = np.argsort(-course_probas)
            total_races += 1
            if course_labels[rankings[0]] == 1:
                top1_correct += 1
            if any(course_labels[rankings[i]] == 1 for i in range(min(3, len(rankings)))):
                top3_correct += 1

        self._save_model()

        metrics = {
            "total_partants": len(y),
            "total_courses_test": total_races,
            "top1_accuracy": round(top1_correct / max(total_races, 1) * 100, 1),
            "top3_accuracy": round(top3_correct / max(total_races, 1) * 100, 1),
            "feature_names": FEATURE_NAMES,
        }

        # Feature importance
        if hasattr(base_model, "feature_importances_"):
            importances = base_model.feature_importances_
            fi = sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1])
            metrics["feature_importance"] = {name: float(imp) for name, imp in fi}

        logger.info("Entraînement terminé : %s", metrics)
        return metrics

    async def predict_course(self, session: AsyncSession, course: Course) -> list[dict]:
        """
        Prédit les probabilités de victoire pour chaque partant d'une course.
        Retourne une liste triée par probabilité décroissante.
        """
        if self.model is None:
            logger.warning("Aucun modèle entraîné, utilisation de la baseline (cote)")
            return await self._baseline_predict(session, course)

        rows = await build_features_for_course(session, course)
        if not rows:
            return []

        X = np.array([r["features"] for r in rows])
        probas = self.model.predict_proba(X)[:, 1]

        results = []
        for i, row in enumerate(rows):
            prob = float(probas[i])
            cote = row["features"][0]  # cote_probable
            prob_implicite = 1.0 / max(cote, 1.0)
            is_value = prob > prob_implicite * 1.2  # 20% de marge pour value bet

            results.append({
                "partant_id": row["partant_id"],
                "cheval_id": row["cheval_id"],
                "numero": row["numero"],
                "probabilite": prob,
                "is_value_bet": is_value,
            })

        # Trier par probabilité décroissante
        results.sort(key=lambda x: -x["probabilite"])

        # Attribuer les rangs et scores de confiance
        total_prob = sum(r["probabilite"] for r in results)
        for rank, r in enumerate(results, 1):
            r["rang_predit"] = rank
            # Score de confiance : écart relatif entre le 1er et le 2ème
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

        # Récupérer historique récent
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
        stmt = (
            select(Prediction)
            .where(Prediction.course_id == course.id)
        )
        result = await session.execute(stmt)
        preds = result.scalars().all()

        mise = 1.0

        for pred in preds:
            partant_stmt = select(Partant).where(Partant.id == pred.partant_id)
            partant = (await session.execute(partant_stmt)).scalar_one_or_none()
            if not partant:
                continue

            # classement None = hors podium (non classé dans les rapports PMU)
            classement = partant.classement
            pred.resultat_gagnant = classement == 1
            pred.resultat_place = classement is not None and 1 <= classement <= 3

            if pred.rang_predit == 1:  # On ne parie que sur notre favori
                if pred.resultat_gagnant and partant.rapport_gagnant:
                    pred.gain_gagnant = partant.rapport_gagnant - mise
                else:
                    pred.gain_gagnant = -mise

                if pred.resultat_place and partant.rapport_place:
                    pred.gain_place = partant.rapport_place - mise
                else:
                    pred.gain_place = -mise

        await session.flush()
