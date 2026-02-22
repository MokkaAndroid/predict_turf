"""
Point d'entrée FastAPI — Application de Prévisions Hippiques.
"""
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
    finally:
        await collector.close()


@app.post("/api/train")
async def train_model(db: AsyncSession = Depends(get_db)):
    """Entraîne le modèle ML sur les données historiques."""
    metrics = await predictor.train(db)
    return {"status": "ok", "metrics": metrics}


@app.post("/api/predict")
async def predict_all(db: AsyncSession = Depends(get_db)):
    """Génère les prédictions pour toutes les courses à venir non encore prédites."""
    stmt = select(Course).where(Course.statut == "A_VENIR")
    result = await db.execute(stmt)
    courses = result.scalars().all()

    count = 0
    for course in courses:
        # Vérifier si des prédictions existent déjà
        pred_check = select(Prediction).where(Prediction.course_id == course.id)
        existing = await db.execute(pred_check)
        if existing.scalars().first():
            continue

        await predictor.predict_and_save(db, course)
        count += 1

    await db.commit()
    return {"status": "ok", "courses_predites": count}


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
