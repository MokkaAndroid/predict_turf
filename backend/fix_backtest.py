"""Recalcule le backtest sur toutes les courses terminees."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.database import init_db, async_session
from app.ml.model import HippiquePredictor
from app.models import Course, Prediction
from sqlalchemy import select, update


async def main():
    await init_db()
    predictor = HippiquePredictor()

    async with async_session() as session:
        # Reset tous les resultats de backtest
        await session.execute(
            update(Prediction).values(
                resultat_gagnant=None,
                resultat_place=None,
                gain_gagnant=None,
                gain_place=None,
            )
        )
        await session.commit()
        print("Reset des resultats de backtest effectue")

        # Relancer le backtest sur toutes les courses terminees
        stmt = select(Course).where(Course.statut == "TERMINE").order_by(Course.date.asc())
        courses = (await session.execute(stmt)).scalars().all()
        print(f"{len(courses)} courses a backtester...")

        for i, course in enumerate(courses):
            await predictor.backtest_course(session, course)
            if (i + 1) % 100 == 0:
                await session.commit()
                print(f"  {i+1}/{len(courses)}...")

        await session.commit()

        # Stats
        pred_bt = (await session.execute(
            select(Prediction).where(
                Prediction.rang_predit == 1,
                Prediction.resultat_gagnant.is_not(None),
            )
        )).scalars().all()

        n = len(pred_bt)
        if n:
            g = sum(1 for p in pred_bt if p.resultat_gagnant)
            pl = sum(1 for p in pred_bt if p.resultat_place)
            pg = sum(p.gain_gagnant or 0 for p in pred_bt)
            pp = sum(p.gain_place or 0 for p in pred_bt)
            print(f"\n=== RESULTATS ({n} courses) ===")
            print(f"  Gagnant : {g}/{n} ({g/n*100:.1f}%)")
            print(f"  Place   : {pl}/{n} ({pl/n*100:.1f}%)")
            print(f"  P&L G   : {pg:+.2f}E  (ROI {pg/n*100:+.1f}%)")
            print(f"  P&L P   : {pp:+.2f}E  (ROI {pp/n*100:+.1f}%)")

    print("\nTermine.")


if __name__ == "__main__":
    asyncio.run(main())
