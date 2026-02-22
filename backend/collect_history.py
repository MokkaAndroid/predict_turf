"""Collecte 3 mois d'historique de courses de plat."""
import asyncio
import sys
import os
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from app.database import init_db, async_session
from app.collectors.pmu import PMUCollector
from app.models import Course, Partant, Cheval
from sqlalchemy import select, func


async def main():
    await init_db()

    collector = PMUCollector()
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=90)

    print(f"=== Collecte du {start} au {end} (90 jours) ===")
    total = await collector.collect_range(start, end, only_plat=True)
    await collector.close()

    async with async_session() as session:
        total_courses = (await session.execute(select(func.count(Course.id)))).scalar()
        terminees = (await session.execute(
            select(func.count(Course.id)).where(Course.statut == "TERMINE")
        )).scalar()
        total_partants = (await session.execute(select(func.count(Partant.id)))).scalar()
        total_chevaux = (await session.execute(select(func.count(Cheval.id)))).scalar()

    print(f"\n=== COLLECTE TERMINEE ===")
    print(f"  Courses totales   : {total_courses}")
    print(f"  Courses terminees : {terminees}")
    print(f"  Partants          : {total_partants}")
    print(f"  Chevaux uniques   : {total_chevaux}")


if __name__ == "__main__":
    asyncio.run(main())
