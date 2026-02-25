"""
Point d'entrée FastAPI — Application de Prévisions Hippiques.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import init_db, get_db, async_session
from app.api.courses import router as courses_router
from app.collectors.pmu import PMUCollector
from app.ml.model import HippiquePredictor
from app.models import Course, Prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── État global pour les tâches longues en arrière-plan ────────
_train_status: dict = {"state": "idle"}
_predict_status: dict = {"state": "idle"}


async def mark_top5_confiance(db: AsyncSession, target_date: date):
    """Marque les 5 prédictions rang 1 avec la meilleure confiance pour un jour donné.
    Réinitialise d'abord tous les flags du jour, puis active les 5 meilleurs."""
    from datetime import datetime as dt, timedelta as td
    day_start = dt.combine(target_date, dt.min.time())
    day_end = dt.combine(target_date + td(days=1), dt.min.time())

    # Récupérer toutes les courses du jour
    stmt = select(Course).where(Course.date >= day_start, Course.date < day_end)
    result = await db.execute(stmt)
    courses = result.scalars().all()
    course_ids = [c.id for c in courses]

    if not course_ids:
        return

    # Reset tous les flags du jour
    from sqlalchemy import update
    await db.execute(
        update(Prediction)
        .where(Prediction.course_id.in_(course_ids))
        .values(top5_confiance=False)
    )

    # Récupérer les prédictions rang 1 du jour, triées par confiance
    pred_stmt = (
        select(Prediction)
        .where(
            Prediction.course_id.in_(course_ids),
            Prediction.rang_predit == 1,
        )
        .order_by(Prediction.score_confiance.desc())
        .limit(5)
    )
    top5 = (await db.execute(pred_stmt)).scalars().all()
    for pred in top5:
        pred.top5_confiance = True

    await db.flush()
    logger.info("Top 5 confiance marquées pour le %s (%d prédictions)", target_date, len(top5))

predictor = HippiquePredictor()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    logger.info("Base de données initialisée")
    yield


app = FastAPI(
    title="Prévisions Hippiques",
    description="API de prédiction et backtesting pour les courses de plat (PMU)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(courses_router)


# ── Endpoints d'administration / data pipeline ───────────────────

@app.post("/api/collect")
async def collect_data(
    start: str = Query(..., description="Date début YYYY-MM-DD"),
    end: str = Query(..., description="Date fin YYYY-MM-DD"),
):
    """Lance la collecte des données PMU sur une plage de dates."""
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    collector = PMUCollector()
    try:
        count = await collector.collect_range(start_date, end_date, only_plat=True)
        return {"status": "ok", "courses_collectees": count}
    except Exception as e:
        logger.error("Erreur collecte: %s", e)
        return {"status": "partial", "error": str(e)}
    finally:
        await collector.close()


@app.post("/api/collect/today")
async def collect_today():
    """Collecte les courses du jour."""
    today = date.today()
    collector = PMUCollector()
    try:
        count = await collector.collect_date(today, only_plat=True)
        return {"status": "ok", "courses_collectees": count}
    except Exception as e:
        logger.error("Erreur collecte today: %s", e)
        return {"status": "error", "error": str(e)}
    finally:
        await collector.close()


@app.post("/api/daily-update")
async def daily_update(db: AsyncSession = Depends(get_db)):
    """Mise à jour quotidienne : collecte/rafraîchit les courses du jour,
    supprime les anciennes prédictions du jour, puis re-génère les prédictions."""
    today = date.today()
    collector = PMUCollector()
    try:
        # 1. Collecter les nouvelles courses + mettre à jour les existantes
        update_result = await collector.update_today(today, only_plat=True)
    except Exception as e:
        logger.error("Erreur daily-update collecte: %s", e)
        return {"status": "error", "step": "collect", "error": str(e)}
    finally:
        await collector.close()

    # 2. Supprimer les prédictions existantes du jour pour les re-générer
    from datetime import datetime as dt, timedelta as td
    today_start = dt.combine(today, dt.min.time())
    tomorrow_start = dt.combine(today + td(days=1), dt.min.time())

    stmt = select(Course).where(
        Course.date >= today_start,
        Course.date < tomorrow_start,
        Course.statut == "A_VENIR",
    )
    result = await db.execute(stmt)
    courses_today = result.scalars().all()

    deleted_preds = 0
    for course in courses_today:
        del_stmt = select(Prediction).where(Prediction.course_id == course.id)
        existing_preds = await db.execute(del_stmt)
        for pred in existing_preds.scalars().all():
            await db.delete(pred)
            deleted_preds += 1
    await db.commit()

    # 3. Re-générer les prédictions
    predicted = 0
    errors = 0
    if predictor.model is not None:
        for course in courses_today:
            try:
                await predictor.predict_and_save(db, course)
                predicted += 1
            except Exception as e:
                logger.error("Erreur prediction course %s: %s", course.id, e)
                errors += 1
        await db.commit()

        # 4. Marquer les top 5 confiance du jour
        await mark_top5_confiance(db, today)
        await db.commit()

    return {
        "status": "ok",
        "courses_mises_a_jour": update_result.get("updated", 0),
        "courses_creees": update_result.get("created", 0),
        "predictions_supprimees": deleted_preds,
        "predictions_generees": predicted,
        "erreurs_prediction": errors,
        "model_loaded": predictor.model is not None,
    }


@app.post("/api/train")
async def train_model(
    train_end: str = Query("2026-01-31", description="Dernière date d'entraînement (YYYY-MM-DD)"),
    test_start: str = Query("2026-02-01", description="Première date du test set (YYYY-MM-DD)"),
):
    """Lance l'entraînement du modèle ML en arrière-plan avec split temporel."""
    global _train_status
    if _train_status.get("state") == "running":
        return {"status": "already_running"}

    train_end_date = date.fromisoformat(train_end)
    test_start_date = date.fromisoformat(test_start)

    _train_status = {"state": "running"}

    async def _do_train():
        global _train_status
        try:
            async with async_session() as db:
                metrics = await predictor.train(db, train_end=train_end_date, test_start=test_start_date)
            _train_status = {"state": "done", "result": {"status": "ok", "metrics": metrics}}
        except Exception as e:
            logger.error("Erreur entraînement en arrière-plan: %s", e)
            _train_status = {"state": "done", "result": {"status": "error", "error": str(e)}}

    asyncio.create_task(_do_train())
    return {"status": "started"}


@app.get("/api/train/status")
async def train_status():
    """Retourne l'état de l'entraînement en cours."""
    return _train_status


@app.post("/api/tune")
async def tune_model(
    n_trials: int = Query(50, description="Nombre d'essais Optuna"),
    db: AsyncSession = Depends(get_db),
):
    """Lance le tuning des hyperparamètres avec Optuna."""
    result = await predictor.tune(db, n_trials=n_trials)
    return {"status": "ok", **result}


@app.post("/api/predict")
async def predict_all():
    """Lance la génération des prédictions en arrière-plan."""
    global _predict_status
    if _predict_status.get("state") == "running":
        return {"status": "already_running"}

    _predict_status = {"state": "running"}

    async def _do_predict():
        global _predict_status
        try:
            async with async_session() as db:
                stmt = select(Course).where(Course.statut == "A_VENIR")
                result = await db.execute(stmt)
                courses = result.scalars().all()

                count = 0
                errors = 0
                for course in courses:
                    pred_check = select(Prediction).where(Prediction.course_id == course.id)
                    existing = await db.execute(pred_check)
                    if existing.scalars().first():
                        continue

                    try:
                        await predictor.predict_and_save(db, course)
                        count += 1
                    except Exception as e:
                        logger.error("Erreur prediction course %s: %s", course.id, e)
                        errors += 1

                    if count % 50 == 0 and count > 0:
                        await db.commit()

                await db.commit()

                # Marquer les top 5 confiance pour chaque jour concerné
                jours = set()
                for course in courses:
                    jours.add(course.date.date())
                for jour in jours:
                    await mark_top5_confiance(db, jour)
                await db.commit()

            _predict_status = {"state": "done", "result": {"status": "ok", "courses_predites": count, "erreurs": errors}}
        except Exception as e:
            logger.error("Erreur prédictions en arrière-plan: %s", e)
            _predict_status = {"state": "done", "result": {"status": "error", "error": str(e)}}

    asyncio.create_task(_do_predict())
    return {"status": "started"}


@app.get("/api/predict/status")
async def predict_status():
    """Retourne l'état de la génération de prédictions."""
    return _predict_status


@app.post("/api/backtest")
async def backtest_all(db: AsyncSession = Depends(get_db)):
    """Met à jour le backtesting pour toutes les courses terminées avec prédictions."""
    stmt = (
        select(Course)
        .where(Course.statut == "TERMINE")
    )
    result = await db.execute(stmt)
    courses = result.scalars().all()

    count = 0
    for course in courses:
        # Vérifier s'il y a des prédictions sans résultat de backtest
        pred_stmt = select(Prediction).where(
            Prediction.course_id == course.id,
            Prediction.resultat_gagnant.is_(None),
        )
        preds = await db.execute(pred_stmt)
        if preds.scalars().first():
            await predictor.backtest_course(db, course)
            count += 1

    await db.commit()
    return {"status": "ok", "courses_backtestees": count}


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": predictor.model is not None}


@app.post("/api/reset")
async def reset_database():
    """Supprime toutes les données et recrée les tables."""
    from app.database import engine, Base
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    return {"status": "ok", "message": "Base de données réinitialisée"}


# ── Serve frontend static files (production) ────────────────────
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the React SPA for all non-API routes."""
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")
