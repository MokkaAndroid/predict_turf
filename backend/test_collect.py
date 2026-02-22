"""
Test rapide : collecte les courses du jour et affiche un résumé.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.database import init_db, async_session
from app.collectors.pmu import PMUCollector
from app.models import Course, Partant, Cheval
from sqlalchemy import select, func
from datetime import date, timedelta


async def main():
    print("=== Initialisation de la base de données ===")
    await init_db()

    collector = PMUCollector()

    # Collecter aujourd'hui
    today = date.today()
    print(f"\n=== Collecte des courses de plat du {today} ===")
    n = await collector.collect_date(today, only_plat=True)
    print(f"  -> {n} courses de plat collectées")

    # Collecter hier aussi pour avoir des résultats
    yesterday = today - timedelta(days=1)
    print(f"\n=== Collecte des courses de plat du {yesterday} ===")
    n2 = await collector.collect_date(yesterday, only_plat=True)
    print(f"  -> {n2} courses de plat collectées")

    await collector.close()

    # Résumé
    async with async_session() as session:
        total_courses = (await session.execute(select(func.count(Course.id)))).scalar()
        courses_terminees = (await session.execute(
            select(func.count(Course.id)).where(Course.statut == "TERMINE")
        )).scalar()
        courses_avenir = (await session.execute(
            select(func.count(Course.id)).where(Course.statut == "A_VENIR")
        )).scalar()
        total_partants = (await session.execute(select(func.count(Partant.id)))).scalar()
        total_chevaux = (await session.execute(select(func.count(Cheval.id)))).scalar()

        print("\n=== RÉSUMÉ BASE DE DONNÉES ===")
        print(f"  Courses totales    : {total_courses}")
        print(f"  Courses terminées  : {courses_terminees}")
        print(f"  Courses à venir    : {courses_avenir}")
        print(f"  Partants           : {total_partants}")
        print(f"  Chevaux uniques    : {total_chevaux}")

        # Afficher quelques courses
        stmt = select(Course).order_by(Course.date.desc()).limit(5)
        result = await session.execute(stmt)
        courses = result.scalars().all()
        print("\n=== 5 DERNIÈRES COURSES ===")
        for c in courses:
            partants_stmt = select(func.count(Partant.id)).where(Partant.course_id == c.id)
            nb_p = (await session.execute(partants_stmt)).scalar()
            print(f"  {c.date.strftime('%d/%m %H:%M')} | {c.hippodrome:15s} | R{c.numero_reunion}C{c.numero_course} | {c.distance}m | {nb_p} partants | {c.statut}")

        # Afficher les résultats d'une course terminée
        stmt_term = select(Course).where(Course.statut == "TERMINE").limit(1)
        result_term = await session.execute(stmt_term)
        course_term = result_term.scalar_one_or_none()
        if course_term:
            print(f"\n=== DÉTAIL COURSE TERMINÉE : {course_term.hippodrome} R{course_term.numero_reunion}C{course_term.numero_course} ===")
            stmt_p = select(Partant).where(Partant.course_id == course_term.id).order_by(Partant.classement.asc().nullslast())
            result_p = await session.execute(stmt_p)
            partants = result_p.scalars().all()
            for p in partants:
                cheval = (await session.execute(select(Cheval).where(Cheval.id == p.cheval_id))).scalar_one_or_none()
                nom = cheval.nom if cheval else "?"
                class_str = f"#{p.classement}" if p.classement else "NC"
                rg = f"rapp.G={p.rapport_gagnant:.2f}€" if p.rapport_gagnant else ""
                rp = f"rapp.P={p.rapport_place:.2f}€" if p.rapport_place else ""
                print(f"  N°{p.numero:2d} | {nom:20s} | cote={p.cote_depart or p.cote_probable or '?':>5} | {class_str:>4s} | {rg} {rp}")

    print("\n=== TEST TERMINÉ AVEC SUCCÈS ===")


if __name__ == "__main__":
    asyncio.run(main())
