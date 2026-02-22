"""
Test : génère des prédictions pour les courses à venir (mode baseline sans modèle ML).
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.database import async_session
from app.ml.model import HippiquePredictor
from app.models import Course, Prediction, Partant, Cheval
from sqlalchemy import select


async def main():
    predictor = HippiquePredictor()

    async with async_session() as session:
        # Prédire les courses à venir
        stmt = select(Course).where(Course.statut == "A_VENIR").order_by(Course.date.asc())
        result = await session.execute(stmt)
        courses = result.scalars().all()

        print(f"=== {len(courses)} courses à venir ===\n")

        for course in courses:
            await predictor.predict_and_save(session, course)

            # Afficher les prédictions
            pred_stmt = (
                select(Prediction)
                .where(Prediction.course_id == course.id)
                .order_by(Prediction.rang_predit)
            )
            preds = (await session.execute(pred_stmt)).scalars().all()

            print(f"--- {course.hippodrome} R{course.numero_reunion}C{course.numero_course} | {course.distance}m | {course.date.strftime('%H:%M')} ---")
            for p in preds:
                partant = (await session.execute(select(Partant).where(Partant.id == p.partant_id))).scalar_one()
                cheval = (await session.execute(select(Cheval).where(Cheval.id == partant.cheval_id))).scalar_one()
                vb = " [VALUE BET]" if p.is_value_bet else ""
                print(f"  #{p.rang_predit} {cheval.nom:20s} | N°{partant.numero:2d} | proba={p.probabilite:.1%} | confiance={p.score_confiance:.0f}%{vb}")
                if p.commentaire:
                    print(f"     > {p.commentaire}")
            print()

        await session.commit()

        # Backtester les courses terminées
        print("=== BACKTESTING courses terminées ===\n")
        stmt_term = select(Course).where(Course.statut == "TERMINE").order_by(Course.date.asc())
        courses_term = (await session.execute(stmt_term)).scalars().all()

        for course in courses_term:
            await predictor.predict_and_save(session, course)
        await session.flush()

        for course in courses_term:
            await predictor.backtest_course(session, course)

        await session.commit()

        # Résumé backtesting
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

            print(f"Courses backtestées : {len(pred_bt)}")
            print(f"Gagnant correct     : {gagnants}/{len(pred_bt)} ({gagnants/len(pred_bt)*100:.1f}%)")
            print(f"Placé correct       : {places}/{len(pred_bt)} ({places/len(pred_bt)*100:.1f}%)")
            print(f"P&L gagnant (1€)    : {profit_g:+.2f}€")
            print(f"P&L placé (1€)      : {profit_p:+.2f}€")
            print(f"ROI gagnant         : {profit_g/len(pred_bt)*100:+.1f}%")
            print(f"ROI placé           : {profit_p/len(pred_bt)*100:+.1f}%")
        else:
            print("Aucune prédiction backtestée")

    print("\n=== TEST PRÉDICTIONS TERMINÉ ===")


if __name__ == "__main__":
    asyncio.run(main())
