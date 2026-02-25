"""
Feature engineering pour la prédiction hippique (Plat).
Chaque partant dans une course est transformé en vecteur de features.
39 features au total :
  14 corrigées + 8 DB + 3 tendance + 4 relatives + 1 Equidia
  + 4 interactions + 3 forme avancée + 2 hippodrome = 39
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
    # ── Interactions (4) ──
    "jockey_cheval_winrate",      # win rate jockey sur ce cheval
    "jockey_hippodrome_winrate",  # win rate jockey sur cet hippodrome
    "entraineur_hippodrome_winrate",  # win rate entraîneur sur cet hippodrome
    "jockey_distance_winrate",    # win rate jockey sur cette distance (±200m)
    # ── Forme avancée (3) ──
    "forme_ponderee",       # moyenne pondérée classements récents (dernière = poids 5)
    "forme_tendance",       # pente de la forme (négatif = progression)
    "nb_victoires_recentes",  # victoires sur les 5 dernières courses
    # ── Hippodrome (2) ──
    "hippodrome_favori_winrate",  # target encoding: win rate du favori cote sur cet hippodrome
    "cheval_hippodrome_winrate",  # win rate du cheval sur cet hippodrome
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

    # Pré-calculer le hippodrome_favori_winrate (target encoding)
    hippo_favori_wr = await _hippodrome_favori_winrate(session, course)

    rows = []
    for p in partants:
        features = await _compute_features(session, p, course, hippo_favori_wr)
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


async def _hippodrome_favori_winrate(session: AsyncSession, course: Course) -> float:
    """Target encoding : taux de victoire du favori (cote la plus basse) sur cet hippodrome."""
    if not course.hippodrome:
        return float("nan")

    # Nombre de courses terminées sur cet hippodrome avant cette course
    total_stmt = (
        select(func.count(Course.id))
        .where(
            Course.hippodrome == course.hippodrome,
            Course.statut == "TERMINE",
            Course.date < course.date,
        )
    )
    total = (await session.execute(total_stmt)).scalar() or 0
    if total < 5:
        return float("nan")

    # Courses où le favori (cote min) a gagné
    # On utilise une sous-requête: pour chaque course terminée sur cet hippo,
    # on vérifie si le partant avec la plus petite cote_probable a classement=1
    past_courses_stmt = (
        select(Course.id)
        .where(
            Course.hippodrome == course.hippodrome,
            Course.statut == "TERMINE",
            Course.date < course.date,
        )
    )
    past_result = await session.execute(past_courses_stmt)
    past_course_ids = [r[0] for r in past_result.fetchall()]

    if not past_course_ids:
        return float("nan")

    favori_wins = 0
    # Batch: pour chaque course passée, trouver le favori et vérifier s'il a gagné
    for cid in past_course_ids[-100:]:  # Limiter aux 100 dernières pour perf
        fav_stmt = (
            select(Partant)
            .where(
                Partant.course_id == cid,
                Partant.statut == "PARTANT",
                Partant.cote_probable.isnot(None),
            )
            .order_by(Partant.cote_probable.asc())
            .limit(1)
        )
        fav = (await session.execute(fav_stmt)).scalar_one_or_none()
        if fav and fav.classement == 1:
            favori_wins += 1

    return favori_wins / len(past_course_ids[-100:])


def _add_relative_features(rows: list[dict]):
    """Calcule les 4 features relatives à la course et les ajoute au vecteur."""
    if not rows:
        return

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

        # Pop the last element (rang_prono_equidia), insert relatives, re-add prono
        prono = f.pop()
        f.extend([rang_cote, ecart_cote_favori, win_rate_relatif, poids_relatif])
        f.append(prono)


async def _compute_features(
    session: AsyncSession, partant: Partant, course: Course,
    hippo_favori_wr: float,
) -> list[float]:
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

    # ── Forme avancée (Point 2) ───────────────────────────
    # Forme pondérée : dernière course poids 5, avant-dernière poids 4, etc.
    forme_ponderee = float("nan")
    forme_tendance = float("nan")
    nb_victoires_recentes = 0.0
    if classements_recents:
        n = len(classements_recents)
        poids_w = list(range(n, 0, -1))  # [5,4,3,2,1] pour 5 courses
        somme_ponderee = sum(c * w for c, w in zip(classements_recents, poids_w))
        forme_ponderee = somme_ponderee / sum(poids_w)

        nb_victoires_recentes = float(sum(1 for h in recents if h.classement == 1))

        # Tendance : pente via régression linéaire simple (y = classements, x = 0,1,2...)
        # Pente négative = progression (classements en baisse = mieux)
        if n >= 2:
            # classements_recents[0] = plus récent, classements_recents[-1] = plus ancien
            # On inverse pour avoir x=0 = plus ancien
            y = list(reversed(classements_recents))
            x_mean = (n - 1) / 2
            y_mean = sum(y) / n
            num = sum((i - x_mean) * (yi - y_mean) for i, yi in enumerate(y))
            den = sum((i - x_mean) ** 2 for i in range(n))
            forme_tendance = num / den if den > 0 else 0.0

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

    # ── Interactions Jockey×Cheval (Point 1) ──────────────
    jockey_cheval_wr = float("nan")
    if partant.jockey_id and partant.cheval_id:
        jc_total_stmt = select(func.count(Partant.id)).join(Course).where(
            Partant.jockey_id == partant.jockey_id,
            Partant.cheval_id == partant.cheval_id,
            Course.statut == "TERMINE",
            Course.date < course.date,
        )
        jc_total = (await session.execute(jc_total_stmt)).scalar() or 0
        if jc_total > 0:
            jc_wins_stmt = select(func.count(Partant.id)).join(Course).where(
                Partant.jockey_id == partant.jockey_id,
                Partant.cheval_id == partant.cheval_id,
                Partant.classement == 1,
                Course.statut == "TERMINE",
                Course.date < course.date,
            )
            jc_wins = (await session.execute(jc_wins_stmt)).scalar() or 0
            jockey_cheval_wr = jc_wins / jc_total

    # ── Interactions Jockey×Hippodrome (Point 1) ──────────
    jockey_hippo_wr = float("nan")
    if partant.jockey_id and course.hippodrome:
        jh_total_stmt = select(func.count(Partant.id)).join(Course).where(
            Partant.jockey_id == partant.jockey_id,
            Course.hippodrome == course.hippodrome,
            Course.statut == "TERMINE",
            Course.date < course.date,
        )
        jh_total = (await session.execute(jh_total_stmt)).scalar() or 0
        if jh_total >= 3:  # minimum 3 courses pour être significatif
            jh_wins_stmt = select(func.count(Partant.id)).join(Course).where(
                Partant.jockey_id == partant.jockey_id,
                Partant.classement == 1,
                Course.hippodrome == course.hippodrome,
                Course.statut == "TERMINE",
                Course.date < course.date,
            )
            jh_wins = (await session.execute(jh_wins_stmt)).scalar() or 0
            jockey_hippo_wr = jh_wins / jh_total

    # ── Interactions Entraîneur×Hippodrome (Point 1) ──────
    entr_hippo_wr = float("nan")
    if partant.entraineur_id and course.hippodrome:
        eh_total_stmt = select(func.count(Partant.id)).join(Course).where(
            Partant.entraineur_id == partant.entraineur_id,
            Course.hippodrome == course.hippodrome,
            Course.statut == "TERMINE",
            Course.date < course.date,
        )
        eh_total = (await session.execute(eh_total_stmt)).scalar() or 0
        if eh_total >= 3:
            eh_wins_stmt = select(func.count(Partant.id)).join(Course).where(
                Partant.entraineur_id == partant.entraineur_id,
                Partant.classement == 1,
                Course.hippodrome == course.hippodrome,
                Course.statut == "TERMINE",
                Course.date < course.date,
            )
            eh_wins = (await session.execute(eh_wins_stmt)).scalar() or 0
            entr_hippo_wr = eh_wins / eh_total

    # ── Interactions Jockey×Distance (Point 1) ────────────
    jockey_dist_wr = float("nan")
    if partant.jockey_id and course.distance:
        d_min = course.distance - 200
        d_max = course.distance + 200
        jd_total_stmt = select(func.count(Partant.id)).join(Course).where(
            Partant.jockey_id == partant.jockey_id,
            Course.statut == "TERMINE",
            Course.date < course.date,
            Course.distance.between(d_min, d_max),
        )
        jd_total = (await session.execute(jd_total_stmt)).scalar() or 0
        if jd_total >= 3:
            jd_wins_stmt = select(func.count(Partant.id)).join(Course).where(
                Partant.jockey_id == partant.jockey_id,
                Partant.classement == 1,
                Course.statut == "TERMINE",
                Course.date < course.date,
                Course.distance.between(d_min, d_max),
            )
            jd_wins = (await session.execute(jd_wins_stmt)).scalar() or 0
            jockey_dist_wr = jd_wins / jd_total

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

    # ── Cheval×Hippodrome win rate (Point 3) ──────────────
    cheval_hippo_wr = float("nan")
    if partant.cheval_id and course.hippodrome:
        ch_total_stmt = (
            select(func.count(Partant.id))
            .join(Course)
            .where(
                Partant.cheval_id == partant.cheval_id,
                Course.hippodrome == course.hippodrome,
                Course.statut == "TERMINE",
                Course.date < course.date,
            )
        )
        ch_total = (await session.execute(ch_total_stmt)).scalar() or 0
        if ch_total > 0:
            ch_wins_stmt = (
                select(func.count(Partant.id))
                .join(Course)
                .where(
                    Partant.cheval_id == partant.cheval_id,
                    Partant.classement == 1,
                    Course.hippodrome == course.hippodrome,
                    Course.statut == "TERMINE",
                    Course.date < course.date,
                )
            )
            ch_wins = (await session.execute(ch_wins_stmt)).scalar() or 0
            cheval_hippo_wr = ch_wins / ch_total

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
        # ── Interactions (4) ──
        jockey_cheval_wr,
        jockey_hippo_wr,
        entr_hippo_wr,
        jockey_dist_wr,
        # ── Forme avancée (3) ──
        forme_ponderee,
        forme_tendance,
        nb_victoires_recentes,
        # ── Hippodrome (2) ──
        hippo_favori_wr,
        cheval_hippo_wr,
        # ── Relatives (4) seront insérées par _add_relative_features ──
        # ── Pronostics (1, sera déplacé à la fin par _add_relative_features) ──
        rang_prono,
    ]
