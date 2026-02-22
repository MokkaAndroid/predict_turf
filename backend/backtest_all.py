"""Backtest complet : genere des predictions et backteste toutes les courses terminees."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.database import init_db, async_session
from app.ml.model import HippiquePredictor
from app.models import Course, Prediction
from sqlalchemy import select, func


async def main():
    await init_db()
    predictor = HippiquePredictor()

    async with async_session() as session:
        # Toutes les courses terminees sans prediction
        stmt = (
            select(Course)
            .where(Course.statut == "TERMINE")
            .order_by(Course.date.asc())
        )
        result = await session.execute(stmt)
        courses = result.scalars().all()

        total = len(courses)
        print(f"=== {total} courses terminees a traiter ===")

        for i, course in enumerate(courses):
            # Skip si deja predit
            existing = await session.execute(
                select(func.count(Prediction.id)).where(Prediction.course_id == course.id)
            )
            if existing.scalar() > 0:
                continue

            await predictor.predict_and_save(session, course)
            await predictor.backtest_course(session, course)

            if (i + 1) % 50 == 0:
                await session.commit()
                print(f"  {i+1}/{total} courses traitees...")

        await session.commit()

        # Stats finales
        pred_bt = (await session.execute(
            select(Prediction).where(
                Prediction.rang_predit == 1,
                Prediction.resultat_gagnant.is_not(None),
            )
        )).scalars().all()

        if pred_bt:
            gagnants = sum(1 for p in pred_bt if p.resultat_gagnant)
            places = sum(1 for p in pred_bt if p.resultat_place)
            profit_g = sum(p.gain_gagnant or 0 for p in pred_bt)
            profit_p = sum(p.gain_place or 0 for p in pred_bt)
            n = len(pred_bt)

            print(f"\n=== RESULTATS BACKTESTING ({n} courses) ===")
            print(f"  Gagnant correct  : {gagnants}/{n} ({gagnants/n*100:.1f}%)")
            print(f"  Place correct    : {places}/{n} ({places/n*100:.1f}%)")
            print(f"  P&L Gagnant (1E) : {profit_g:+.2f}E")
            print(f"  P&L Place (1E)   : {profit_p:+.2f}E")
            print(f"  ROI Gagnant      : {profit_g/n*100:+.1f}%")
            print(f"  ROI Place        : {profit_p/n*100:+.1f}%")

    print("\n=== BACKTEST TERMINE ===")


if __name__ == "__main__":
    asyncio.run(main())
