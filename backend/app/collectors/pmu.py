"""
Collecteur de données PMU via l'API publique turfinfo.
Endpoints utilisés :
  - /programme/{DDMMYYYY}                     → réunions et courses du jour
  - /programme/{date}/R{n}/C{n}/participants   → partants détaillés
  - /programme/{date}/R{n}/C{n}/pronostics     → pronostics Equidia
  - /programme/{date}/R{n}/C{n}/rapports-definitifs → résultats et rapports
"""
import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session
from app.models import Course, Cheval, Jockey, Entraineur, Partant

logger = logging.getLogger(__name__)

BASE = settings.pmu_base_url


def _date_fmt(d: date) -> str:
    """Format date as DDMMYYYY for PMU API."""
    return d.strftime("%d%m%Y")


class PMUCollector:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30, headers={"Accept": "application/json"})

    async def close(self):
        await self.client.aclose()

    # ── API Calls ─────────────────────────────────────────────

    async def get_programme(self, d: date) -> dict | None:
        url = f"{BASE}/programme/{_date_fmt(d)}"
        resp = await self.client.get(url)
        if resp.status_code != 200:
            logger.warning("Programme %s : HTTP %s", d, resp.status_code)
            return None
        data = resp.json()
        if "programme" not in data:
            return None
        return data["programme"]

    async def get_participants(self, d: date, reunion: int, course: int) -> list[dict]:
        url = f"{BASE}/programme/{_date_fmt(d)}/R{reunion}/C{course}/participants"
        resp = await self.client.get(url)
        if resp.status_code != 200:
            return []
        data = resp.json()
        return data.get("participants", [])

    async def get_rapports(self, d: date, reunion: int, course: int) -> list[dict]:
        url = f"{BASE}/programme/{_date_fmt(d)}/R{reunion}/C{course}/rapports-definitifs"
        resp = await self.client.get(url)
        if resp.status_code != 200:
            return []
        try:
            return resp.json()
        except Exception:
            return []

    # ── Extraction des résultats depuis les rapports ──────────

    @staticmethod
    def parse_rapports(rapports: list[dict]) -> dict[str, Any]:
        """Extrait le gagnant, les placés et leurs rapports depuis rapports-definitifs."""
        result = {"gagnant": None, "places": {}, "rapports_gagnant": {}, "rapports_place": {}}
        for r in rapports:
            if r.get("typePari") == "SIMPLE_GAGNANT" and r.get("rapports"):
                rg = r["rapports"][0]
                num = rg.get("combinaison", "").strip()
                result["gagnant"] = int(num) if num.isdigit() else None
                result["rapports_gagnant"][result["gagnant"]] = rg.get("dividendePourUnEuro", 0) / 100
            if r.get("typePari") == "SIMPLE_PLACE" and r.get("rapports"):
                for rp in r["rapports"]:
                    num = rp.get("combinaison", "").strip()
                    if num.isdigit():
                        n = int(num)
                        result["places"][n] = True
                        result["rapports_place"][n] = rp.get("dividendePourUnEuro", 0) / 100
        return result

    # ── Persist to DB ─────────────────────────────────────────

    async def _get_or_create_cheval(self, session: AsyncSession, p: dict) -> Cheval:
        nom = p.get("nom", "INCONNU")
        stmt = select(Cheval).where(Cheval.nom == nom)
        result = await session.execute(stmt)
        cheval = result.scalar_one_or_none()
        if cheval is None:
            cheval = Cheval(
                nom=nom,
                age=p.get("age"),
                sexe=p.get("sexe"),
                race=p.get("race"),
                pere=p.get("nomPere"),
                mere=p.get("nomMere"),
                proprietaire=p.get("proprietaire"),
            )
            session.add(cheval)
            await session.flush()
        else:
            if p.get("age"):
                cheval.age = p["age"]
        return cheval

    async def _get_or_create_jockey(self, session: AsyncSession, nom: str) -> Jockey:
        stmt = select(Jockey).where(Jockey.nom == nom)
        result = await session.execute(stmt)
        jockey = result.scalar_one_or_none()
        if jockey is None:
            jockey = Jockey(nom=nom)
            session.add(jockey)
            await session.flush()
        return jockey

    async def _get_or_create_entraineur(self, session: AsyncSession, nom: str) -> Entraineur:
        stmt = select(Entraineur).where(Entraineur.nom == nom)
        result = await session.execute(stmt)
        ent = result.scalar_one_or_none()
        if ent is None:
            ent = Entraineur(nom=nom)
            session.add(ent)
            await session.flush()
        return ent

    async def collect_date(self, d: date, only_plat: bool = True) -> int:
        """Collecte toutes les courses d'une date. Retourne le nombre de courses insérées."""
        programme = await self.get_programme(d)
        if not programme:
            logger.info("Aucun programme pour %s", d)
            return 0

        count = 0
        reunions = programme.get("reunions", [])

        async with async_session() as session:
            for reunion in reunions:
                num_reunion = reunion.get("numOfficiel", 0)
                hippodrome_data = reunion.get("hippodrome", {})
                hippodrome = hippodrome_data.get("libelleCourt", "INCONNU")
                pays = reunion.get("pays", {}).get("code", "")

                # On ne prend que les courses françaises
                if pays != "FRA":
                    continue

                courses = reunion.get("courses", [])
                for course_data in courses:
                    discipline = course_data.get("discipline", "")
                    if only_plat and discipline != "PLAT":
                        continue

                    num_course = course_data.get("numOrdre", 0)
                    pmu_id = f"{_date_fmt(d)}_R{num_reunion}_C{num_course}"

                    # Vérifier si la course existe déjà
                    stmt = select(Course).where(Course.pmu_id == pmu_id)
                    result = await session.execute(stmt)
                    existing = result.scalar_one_or_none()
                    if existing:
                        continue

                    # Déterminer la date/heure de départ
                    heure_ts = course_data.get("heureDepart", 0)
                    heure_depart = datetime.fromtimestamp(heure_ts / 1000) if heure_ts else datetime.combine(d, datetime.min.time())

                    course = Course(
                        pmu_id=pmu_id,
                        date=heure_depart,
                        hippodrome=hippodrome,
                        numero_reunion=num_reunion,
                        numero_course=num_course,
                        discipline=discipline,
                        distance=course_data.get("distance"),
                        nombre_partants=course_data.get("nombreDeclaresPartants"),
                        dotation=course_data.get("montantTotalOffert"),
                        categorie=course_data.get("categorieParticularite"),
                        statut="A_VENIR",
                    )
                    session.add(course)
                    await session.flush()

                    # Récupérer les participants
                    participants = await self.get_participants(d, num_reunion, num_course)
                    for p in participants:
                        if p.get("statut") == "NON_PARTANT":
                            continue

                        cheval = await self._get_or_create_cheval(session, p)
                        jockey = await self._get_or_create_jockey(session, p.get("driver", "INCONNU"))
                        entraineur = await self._get_or_create_entraineur(session, p.get("entraineur", "INCONNU"))

                        cote_ref = None
                        if p.get("dernierRapportReference"):
                            cote_ref = p["dernierRapportReference"].get("rapport")

                        cote_direct = None
                        if p.get("dernierRapportDirect"):
                            cote_direct = p["dernierRapportDirect"].get("rapport")

                        partant = Partant(
                            course_id=course.id,
                            cheval_id=cheval.id,
                            jockey_id=jockey.id,
                            entraineur_id=entraineur.id,
                            numero=p.get("numPmu", 0),
                            poids=(p.get("handicapPoids") or 0) / 10,  # centièmes de kg → kg
                            cote_probable=cote_ref,
                            cote_depart=cote_direct,
                        )
                        session.add(partant)

                    # Récupérer les résultats si la course est terminée
                    rapports = await self.get_rapports(d, num_reunion, num_course)
                    if rapports and not isinstance(rapports, dict):  # dict = erreur API
                        parsed = self.parse_rapports(rapports)
                        if parsed["gagnant"]:
                            course.statut = "TERMINE"
                            await session.flush()

                            # Mettre à jour les classements et rapports
                            stmt = select(Partant).where(Partant.course_id == course.id)
                            result = await session.execute(stmt)
                            partants_db = result.scalars().all()

                            for pt in partants_db:
                                if pt.numero == parsed["gagnant"]:
                                    pt.classement = 1
                                    pt.rapport_gagnant = parsed["rapports_gagnant"].get(pt.numero, 0)
                                    pt.rapport_place = parsed["rapports_place"].get(pt.numero, 0)
                                elif pt.numero in parsed["places"]:
                                    pt.classement = 2 if len([x for x in parsed["places"] if x != parsed["gagnant"]]) <= 2 else 3
                                    pt.rapport_place = parsed["rapports_place"].get(pt.numero, 0)

                    count += 1
                    # Throttle pour ne pas surcharger l'API
                    await asyncio.sleep(0.3)

            await session.commit()
        return count

    async def collect_range(self, start: date, end: date, only_plat: bool = True) -> int:
        """Collecte les courses sur une plage de dates."""
        total = 0
        current = start
        while current <= end:
            logger.info("Collecte %s ...", current)
            n = await self.collect_date(current, only_plat=only_plat)
            logger.info("  → %d courses de plat insérées", n)
            total += n
            current += timedelta(days=1)
            await asyncio.sleep(0.5)
        logger.info("Total collecté : %d courses", total)
        return total
