"""
Feature engineering pour la prédiction hippique (Plat).
Chaque partant dans une course est transformé en vecteur de features.
30 features au total : 14 corrigées + 8 DB + 3 tendance + 4 relatives + 1 Equidia.
"""
import logging
import math
from statistics import stdev

import numpy as np
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Partant, Course, Cheval, Jockey, Entraineur

logger = logging.getLogger(__name__)

# ── Mapping catégorie → niveau ordinal ───────────────────────────
_CATEGORIE_LEVELS = {
    "RECLAMER": 0.5,
    "HANDICAP": 1.0,
    "COURSE A CONDITIONS": 2.0,
    "CONDITIONS": 2.0,
    "PRIX": 2.0,
    "QUINTE": 2.5,
    "LISTED": 3.0,
    "LISTEE": 3.0,
    "GROUPE III": 4.0,
    "GROUPE_III": 4.0,
    "GROUPE II": 5.0,
    "GROUPE_II": 5.0,
    "GROUPE I": 6.0,
    "GROUPE_I": 6.0,
}

# ── Mapping condition piste → ordinal ────────────────────────────
_CONDITION_LEVELS = {
    "BON": 0,
    "ASSEZ BON": 0,
    "LEGER": 0,
    "BON LEGER": 0,
    "BON SOUPLE": 1,
    "SOUPLE": 1,
    "ASSEZ SOUPLE": 1,
    "TRES SOUPLE": 2,
    "COLLANT": 2,
    "LOURD": 3,
    "TRES LOURD": 3,
}

FEATURE_NAMES = [
    # ── Originales (14, corrigées) ──
    "cote_probable",
    "cote_depart",
    "poids",
    "nombre_courses",
    "win_rate",
    "place_rate",
    "forme_recente_avg",
    "jours_depuis_derniere",
    "jockey_win_rate",
    "entraineur_win_rate",
    "distance_affinite",
    "gains_carriere_log",
    "nb_partants",
    "prob_implicite_cote",
    # ── Nouvelles DB (8) ──
    "surface",              # HERBE=0, PSF=1
    "condition_piste",      # ordinal 0–3
    "dotation_log",         # log1p(dotation)
    "categorie_level",      # ordinal 0.5–6
    "age",                  # âge du cheval
    "sexe",                 # F=0, M=1, H=2
    "surface_affinite",     # win_rate sur cette surface
    "condition_affinite",   # win_rate sur cette condition
    # ── Tendance & dernière perf (3) ──
    "tendance_cote",        # cote_depart - cote_probable
    "dernier_classement",   # classement dernière course
    "regularite",           # écart-type classements récents
    # ── Relatives intra-course (4) ──
    "rang_cote",            # rang cote dans la course (1=favori)
    "ecart_cote_favori",    # écart avec le favori
    "win_rate_relatif",     # win_rate - moyenne course
    "poids_relatif",        # poids - moyenne course
    # ── Pronostics (1) ──
    "rang_prono_equidia",   # rang dans les pronostics Equidia
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
            "classement": p.classement,
        })

    # Ajouter les features relatives intra-course
    _add_relative_features(rows)

    return rows


def _add_relative_features(rows: list[dict]):
    """Calcule les 4 features relatives à la course et les ajoute au vecteur."""
    if not rows:
        return

    # Indices dans le vecteur de features (avant ajout des relatives)
    # Les features individuelles sont aux indices 0..25, les relatives seront 26..29, prono 30
    idx_cote = 0       # cote_probable
    idx_winrate = 4    # win_rate
    idx_poids = 2      # poids

    # Collecter les valeurs non-NaN
    cotes = []
    win_rates = []
    poids_list = []
    for row in rows:
        f = row["features"]
        c = f[idx_cote]
        if not math.isnan(c):
            cotes.append(c)
        wr = f[idx_winrate]
        if not math.isnan(wr):
            win_rates.append(wr)
        p = f[idx_poids]
        if not math.isnan(p):
            poids_list.append(p)

    # Rang des cotes (1=favori = cote la plus basse)
    sorted_cotes = sorted(cotes)
    mean_winrate = np.mean(win_rates) if win_rates else 0.0
    mean_poids = np.mean(poids_list) if poids_list else float("nan")
    min_cote = sorted_cotes[0] if sorted_cotes else float("nan")

    for row in rows:
        f = row["features"]
        cote = f[idx_cote]
        wr = f[idx_winrate]
        poids = f[idx_poids]

        # rang_cote
        if not math.isnan(cote) and sorted_cotes:
            rang_cote = float(sorted_cotes.index(cote) + 1)
        else:
            rang_cote = float("nan")

        # ecart_cote_favori
        if not math.isnan(cote) and not math.isnan(min_cote):
            ecart_cote_favori = cote - min_cote
        else:
            ecart_cote_favori = float("nan")

        # win_rate_relatif
        if not math.isnan(wr):
            win_rate_relatif = wr - mean_winrate
        else:
            win_rate_relatif = float("nan")

        # poids_relatif
        if not math.isnan(poids) and not math.isnan(mean_poids):
            poids_relatif = poids - mean_poids
        else:
            poids_relatif = float("nan")

        # Append the 4 relative features + prono (already at end)
        # The individual features are 0..25, we insert relatives at 26..29
        # rang_prono_equidia is already at index 25 from _compute_features
        # We need to insert before it
        prono = f.pop()  # remove last (rang_prono_equidia)
        f.extend([rang_cote, ecart_cote_favori, win_rate_relatif, poids_relatif])
        f.append(prono)  # re-add at end


async def _compute_features(session: AsyncSession, partant: Partant, course: Course) -> list[float]:
    """Calcule le vecteur de features individuelles pour un partant donné."""

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
    classements_recents = [h.classement for h in recents if h.classement and h.classement > 0]
    if classements_recents:
        forme_avg = sum(classements_recents) / len(classements_recents)
    else:
        forme_avg = float("nan")

    # Jours depuis dernière course
    jours_depuis = float("nan")
    if historique:
        last_course_stmt = select(Course).where(Course.id == historique[0].course_id)
        lc_result = await session.execute(last_course_stmt)
        last_course = lc_result.scalar_one_or_none()
        if last_course:
            delta = (course.date - last_course.date).days
            jours_depuis = float(max(delta, 0))

    # ── Dernier classement & régularité ───────────────────
    dernier_classement = float("nan")
    regularite = float("nan")
    if classements_recents:
        dernier_classement = float(classements_recents[0])
        if len(classements_recents) >= 2:
            regularite = stdev(classements_recents)

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

    # ── Gains carrière (log1p) ────────────────────────────
    cheval_stmt = select(Cheval).where(Cheval.id == partant.cheval_id)
    cheval = (await session.execute(cheval_stmt)).scalar_one_or_none()
    if cheval and cheval.gains_carriere is not None:
        gains_log = math.log1p(cheval.gains_carriere)
    else:
        gains_log = float("nan")

    # ── Cote et probabilité implicite ─────────────────────
    cote_prob = float(partant.cote_probable) if partant.cote_probable else float("nan")
    cote_dep = float(partant.cote_depart) if partant.cote_depart else cote_prob
    if not math.isnan(cote_prob) and cote_prob >= 1.0:
        prob_implicite = 1.0 / cote_prob
    else:
        prob_implicite = float("nan")

    nb_partants = float(course.nombre_partants) if course.nombre_partants else float("nan")

    # ── Nouvelles features DB ─────────────────────────────
    # Surface: HERBE=0, PSF=1
    surface_val = float("nan")
    if course.surface:
        s = course.surface.upper()
        if "HERBE" in s or "GAZON" in s:
            surface_val = 0.0
        elif "PSF" in s or "FIBRE" in s or "SABLE" in s:
            surface_val = 1.0

    # Condition piste: ordinal
    condition_val = float("nan")
    if course.condition_piste:
        cp_upper = course.condition_piste.upper().strip()
        for key, val in _CONDITION_LEVELS.items():
            if key in cp_upper:
                condition_val = float(val)
                break

    # Dotation log
    dotation_log = float("nan")
    if course.dotation and course.dotation > 0:
        dotation_log = math.log1p(course.dotation)

    # Catégorie level
    categorie_level = float("nan")
    if course.categorie:
        cat_upper = course.categorie.upper().strip()
        for key, val in _CATEGORIE_LEVELS.items():
            if key in cat_upper:
                categorie_level = float(val)
                break

    # Age
    age_val = float("nan")
    if cheval and cheval.age is not None:
        age_val = float(cheval.age)

    # Sexe: F=0, M=1, H=2
    sexe_val = float("nan")
    if cheval and cheval.sexe:
        sexe_map = {"F": 0.0, "M": 1.0, "H": 2.0}
        sexe_val = sexe_map.get(cheval.sexe.upper().strip(), float("nan"))

    # ── Affinité surface ──────────────────────────────────
    surface_aff = float("nan")
    if course.surface and cheval:
        s_total_stmt = (
            select(func.count(Partant.id))
            .join(Course)
            .where(
                Partant.cheval_id == partant.cheval_id,
                Course.statut == "TERMINE",
                Course.date < course.date,
                Course.surface == course.surface,
            )
        )
        s_total = (await session.execute(s_total_stmt)).scalar() or 0
        if s_total > 0:
            s_wins_stmt = (
                select(func.count(Partant.id))
                .join(Course)
                .where(
                    Partant.cheval_id == partant.cheval_id,
                    Partant.classement == 1,
                    Course.statut == "TERMINE",
                    Course.date < course.date,
                    Course.surface == course.surface,
                )
            )
            s_wins = (await session.execute(s_wins_stmt)).scalar() or 0
            surface_aff = s_wins / s_total

    # ── Affinité condition piste ──────────────────────────
    condition_aff = float("nan")
    if course.condition_piste and cheval:
        c_total_stmt = (
            select(func.count(Partant.id))
            .join(Course)
            .where(
                Partant.cheval_id == partant.cheval_id,
                Course.statut == "TERMINE",
                Course.date < course.date,
                Course.condition_piste == course.condition_piste,
            )
        )
        c_total = (await session.execute(c_total_stmt)).scalar() or 0
        if c_total > 0:
            c_wins_stmt = (
                select(func.count(Partant.id))
                .join(Course)
                .where(
                    Partant.cheval_id == partant.cheval_id,
                    Partant.classement == 1,
                    Course.statut == "TERMINE",
                    Course.date < course.date,
                    Course.condition_piste == course.condition_piste,
                )
            )
            c_wins = (await session.execute(c_wins_stmt)).scalar() or 0
            condition_aff = c_wins / c_total

    # ── Tendance cote ─────────────────────────────────────
    tendance_cote = float("nan")
    if not math.isnan(cote_dep) and not math.isnan(cote_prob):
        tendance_cote = cote_dep - cote_prob

    # ── Pronostic Equidia ─────────────────────────────────
    rang_prono = float(partant.rang_pronostic) if partant.rang_pronostic else float("nan")

    poids_val = float(partant.poids) if partant.poids else float("nan")

    return [
        # ── Originales (14) ──
        cote_prob,
        cote_dep,
        poids_val,
        float(nb_courses),
        win_rate,
        place_rate,
        forme_avg,
        jours_depuis,
        jockey_win_rate,
        entr_win_rate,
        distance_aff,
        gains_log,
        nb_partants,
        prob_implicite,
        # ── DB (8) ──
        surface_val,
        condition_val,
        dotation_log,
        categorie_level,
        age_val,
        sexe_val,
        surface_aff,
        condition_aff,
        # ── Tendance (3) ──
        tendance_cote,
        dernier_classement,
        regularite,
        # ── Relatives (4) seront insérées par _add_relative_features ──
        # ── Pronostics (1, sera déplacé à la fin par _add_relative_features) ──
        rang_prono,
    ]
