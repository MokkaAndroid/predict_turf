"""
Feature engineering pour la prédiction hippique (Plat).
Chaque partant dans une course est transformé en vecteur de features.
"""
import logging
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Partant, Course, Cheval, Jockey, Entraineur

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "cote_probable",
    "cote_depart",
    "poids",
    "nombre_courses",
    "win_rate",
    "place_rate",
    "forme_recente_avg",  # classement moyen sur 5 dernières courses
    "jours_depuis_derniere",
    "jockey_win_rate",
    "entraineur_win_rate",
    "distance_affinite",  # win rate du cheval sur cette distance (+/- 200m)
    "gains_carriere_norm",
    "nb_partants",
    "prob_implicite_cote",  # 1 / cote
]


async def build_features_for_course(session: AsyncSession, course: Course) -> list[dict]:
    """
    Construit les features pour chaque partant d'une course.
    Retourne une liste de dicts {partant_id, features: list[float], ...}.
    """
    stmt = select(Partant).where(
        Partant.course_id == course.id,
        Partant.statut == "PARTANT",
    )
    result = await session.execute(stmt)
    partants = result.scalars().all()

    if not partants:
        return []

    rows = []
    for p in partants:
        features = await _compute_features(session, p, course)
        rows.append({
            "partant_id": p.id,
            "cheval_id": p.cheval_id,
            "numero": p.numero,
            "features": features,
            "is_winner": p.classement == 1 if p.classement is not None else None,
            "is_placed": p.classement is not None and 1 <= p.classement <= 3 if p.classement else None,
        })
    return rows


async def _compute_features(session: AsyncSession, partant: Partant, course: Course) -> list[float]:
    """Calcule le vecteur de features pour un partant donné."""

    # ── Stats historiques du cheval ────────────────────────
    hist_stmt = (
        select(Partant)
        .join(Course)
        .where(
            Partant.cheval_id == partant.cheval_id,
            Course.statut == "TERMINE",
            Course.date < course.date,
        )
        .order_by(Course.date.desc())
    )
    hist_result = await session.execute(hist_stmt)
    historique = hist_result.scalars().all()

    nb_courses = len(historique)
    nb_victoires = sum(1 for h in historique if h.classement == 1)
    nb_places = sum(1 for h in historique if h.classement and 1 <= h.classement <= 3)
    win_rate = nb_victoires / max(nb_courses, 1)
    place_rate = nb_places / max(nb_courses, 1)

    # Forme récente (5 dernières courses) — classement moyen
    recents = historique[:5]
    if recents:
        classements = [h.classement for h in recents if h.classement and h.classement > 0]
        forme_avg = sum(classements) / max(len(classements), 1) if classements else 10.0
    else:
        forme_avg = 10.0

    # Jours depuis dernière course
    jours_depuis = 60.0  # défaut
    if historique:
        last_course_stmt = select(Course).where(Course.id == historique[0].course_id)
        lc_result = await session.execute(last_course_stmt)
        last_course = lc_result.scalar_one_or_none()
        if last_course:
            delta = (course.date - last_course.date).days
            jours_depuis = float(max(delta, 0))

    # ── Stats jockey ──────────────────────────────────────
    jockey_win_rate = 0.0
    if partant.jockey_id:
        j_total_stmt = select(func.count(Partant.id)).join(Course).where(
            Partant.jockey_id == partant.jockey_id,
            Course.statut == "TERMINE",
            Course.date < course.date,
        )
        j_total = (await session.execute(j_total_stmt)).scalar() or 0
        j_wins_stmt = select(func.count(Partant.id)).join(Course).where(
            Partant.jockey_id == partant.jockey_id,
            Partant.classement == 1,
            Course.statut == "TERMINE",
            Course.date < course.date,
        )
        j_wins = (await session.execute(j_wins_stmt)).scalar() or 0
        jockey_win_rate = j_wins / max(j_total, 1)

    # ── Stats entraîneur ──────────────────────────────────
    entr_win_rate = 0.0
    if partant.entraineur_id:
        e_total_stmt = select(func.count(Partant.id)).join(Course).where(
            Partant.entraineur_id == partant.entraineur_id,
            Course.statut == "TERMINE",
            Course.date < course.date,
        )
        e_total = (await session.execute(e_total_stmt)).scalar() or 0
        e_wins_stmt = select(func.count(Partant.id)).join(Course).where(
            Partant.entraineur_id == partant.entraineur_id,
            Partant.classement == 1,
            Course.statut == "TERMINE",
            Course.date < course.date,
        )
        e_wins = (await session.execute(e_wins_stmt)).scalar() or 0
        entr_win_rate = e_wins / max(e_total, 1)

    # ── Affinité distance ─────────────────────────────────
    distance_aff = 0.0
    if course.distance:
        d_min = course.distance - 200
        d_max = course.distance + 200
        d_total_stmt = (
            select(func.count(Partant.id))
            .join(Course)
            .where(
                Partant.cheval_id == partant.cheval_id,
                Course.statut == "TERMINE",
                Course.date < course.date,
                Course.distance.between(d_min, d_max),
            )
        )
        d_total = (await session.execute(d_total_stmt)).scalar() or 0
        d_wins_stmt = (
            select(func.count(Partant.id))
            .join(Course)
            .where(
                Partant.cheval_id == partant.cheval_id,
                Partant.classement == 1,
                Course.statut == "TERMINE",
                Course.date < course.date,
                Course.distance.between(d_min, d_max),
            )
        )
        d_wins = (await session.execute(d_wins_stmt)).scalar() or 0
        distance_aff = d_wins / max(d_total, 1)

    # ── Gains carrière normalisés ─────────────────────────
    cheval_stmt = select(Cheval).where(Cheval.id == partant.cheval_id)
    cheval = (await session.execute(cheval_stmt)).scalar_one_or_none()
    gains_norm = 0.0  # TODO: enrichir avec gainsCarriere de l'API

    # ── Cote et probabilité implicite ─────────────────────
    cote_prob = partant.cote_probable or 20.0
    cote_dep = partant.cote_depart or cote_prob
    prob_implicite = 1.0 / max(cote_prob, 1.0)

    nb_partants = float(course.nombre_partants or 10)

    return [
        cote_prob,
        cote_dep,
        partant.poids or 58.0,
        float(nb_courses),
        win_rate,
        place_rate,
        forme_avg,
        jours_depuis,
        jockey_win_rate,
        entr_win_rate,
        distance_aff,
        gains_norm,
        nb_partants,
        prob_implicite,
    ]
