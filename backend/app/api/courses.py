from datetime import datetime, date, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Course, Partant, Prediction, Cheval, Jockey, Entraineur
from app.schemas.course import (
    CourseListSchema,
    CourseDetailSchema,
    PartantSchema,
    PredictionSchema,
    PrevisionJourSchema,
    BacktestingStatsSchema,
    BilanVeilleSchema,
    BilanVeilleSummarySchema,
    ConfianceStatsSchema,
)

router = APIRouter(prefix="/api", tags=["courses"])


@router.get("/courses/passees", response_model=list[CourseListSchema])
async def courses_passees(
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    hippodrome: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    # Requête unique avec LEFT JOIN pour éviter les N+1
    stmt = (
        select(Course, Prediction, Partant, Cheval)
        .outerjoin(Prediction, and_(Prediction.course_id == Course.id, Prediction.rang_predit == 1))
        .outerjoin(Partant, Partant.id == Prediction.partant_id)
        .outerjoin(Cheval, Cheval.id == Partant.cheval_id)
        .where(Course.statut == "TERMINE")
        .order_by(Course.date.desc())
    )
    if hippodrome:
        stmt = stmt.where(Course.hippodrome.ilike(f"%{hippodrome}%"))
    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    rows = result.all()

    items = []
    for c, pred, partant, cheval in rows:
        items.append(CourseListSchema(
            id=c.id,
            pmu_id=c.pmu_id,
            date=c.date,
            hippodrome=c.hippodrome,
            numero_reunion=c.numero_reunion,
            numero_course=c.numero_course,
            discipline=c.discipline,
            distance=c.distance,
            nombre_partants=c.nombre_partants,
            statut=c.statut,
            favori_nom=cheval.nom if cheval else None,
            favori_confiance=pred.score_confiance if pred else None,
            top5_confiance=pred.top5_confiance if pred else False,
            prediction_correcte_gagnant=pred.resultat_gagnant if pred else None,
            prediction_correcte_place=pred.resultat_place if pred else None,
            gain_simule_gagnant=pred.gain_gagnant if pred else None,
            gain_simule_place=pred.gain_place if pred else None,
        ))
    return items


@router.get("/courses/a-venir", response_model=list[CourseListSchema])
async def courses_a_venir(
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Course, Prediction, Partant, Cheval)
        .outerjoin(Prediction, and_(Prediction.course_id == Course.id, Prediction.rang_predit == 1))
        .outerjoin(Partant, Partant.id == Prediction.partant_id)
        .outerjoin(Cheval, Cheval.id == Partant.cheval_id)
        .where(Course.statut == "A_VENIR")
        .order_by(Course.date.asc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    rows = result.all()

    items = []
    for c, pred, partant, cheval in rows:
        items.append(CourseListSchema(
            id=c.id,
            pmu_id=c.pmu_id,
            date=c.date,
            hippodrome=c.hippodrome,
            numero_reunion=c.numero_reunion,
            numero_course=c.numero_course,
            discipline=c.discipline,
            distance=c.distance,
            nombre_partants=c.nombre_partants,
            statut=c.statut,
            favori_nom=cheval.nom if cheval else None,
            favori_confiance=pred.score_confiance if pred else None,
            top5_confiance=pred.top5_confiance if pred else False,
        ))
    return items


@router.get("/course/{course_id}", response_model=CourseDetailSchema)
async def course_detail(course_id: int, db: AsyncSession = Depends(get_db)):
    stmt = select(Course).where(Course.id == course_id)
    result = await db.execute(stmt)
    course = result.scalar_one_or_none()
    if not course:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Course non trouvée")

    # Partants
    p_stmt = select(Partant).where(Partant.course_id == course_id).order_by(Partant.numero)
    p_result = await db.execute(p_stmt)
    partants = p_result.scalars().all()

    partant_schemas = []
    for p in partants:
        cheval_nom = ""
        jockey_nom = None
        entraineur_nom = None

        c_stmt = select(Cheval).where(Cheval.id == p.cheval_id)
        cr = await db.execute(c_stmt)
        cheval = cr.scalar_one_or_none()
        if cheval:
            cheval_nom = cheval.nom

        if p.jockey_id:
            j_stmt = select(Jockey).where(Jockey.id == p.jockey_id)
            jr = await db.execute(j_stmt)
            jockey = jr.scalar_one_or_none()
            if jockey:
                jockey_nom = jockey.nom

        if p.entraineur_id:
            e_stmt = select(Entraineur).where(Entraineur.id == p.entraineur_id)
            er = await db.execute(e_stmt)
            ent = er.scalar_one_or_none()
            if ent:
                entraineur_nom = ent.nom

        partant_schemas.append(PartantSchema(
            id=p.id,
            numero=p.numero,
            cheval_nom=cheval_nom,
            jockey_nom=jockey_nom,
            entraineur_nom=entraineur_nom,
            poids=p.poids,
            cote_probable=p.cote_probable,
            cote_depart=p.cote_depart,
            classement=p.classement,
            rapport_gagnant=p.rapport_gagnant,
            rapport_place=p.rapport_place,
            statut=p.statut,
        ))

    # Prédictions
    pr_stmt = (
        select(Prediction)
        .where(Prediction.course_id == course_id)
        .order_by(Prediction.rang_predit)
    )
    pr_result = await db.execute(pr_stmt)
    preds = pr_result.scalars().all()

    pred_schemas = []
    for pred in preds:
        p_obj = next((p for p in partants if p.id == pred.partant_id), None)
        cheval_nom = ""
        numero = 0
        if p_obj:
            c_stmt = select(Cheval).where(Cheval.id == p_obj.cheval_id)
            cr = await db.execute(c_stmt)
            cheval = cr.scalar_one_or_none()
            cheval_nom = cheval.nom if cheval else ""
            numero = p_obj.numero

        pred_schemas.append(PredictionSchema(
            partant_id=pred.partant_id,
            cheval_nom=cheval_nom,
            numero=numero,
            probabilite=pred.probabilite,
            rang_predit=pred.rang_predit,
            score_confiance=pred.score_confiance,
            is_value_bet=pred.is_value_bet,
            commentaire=pred.commentaire,
        ))

    return CourseDetailSchema(
        id=course.id,
        pmu_id=course.pmu_id,
        date=course.date,
        hippodrome=course.hippodrome,
        numero_reunion=course.numero_reunion,
        numero_course=course.numero_course,
        discipline=course.discipline,
        surface=course.surface,
        distance=course.distance,
        condition_piste=course.condition_piste,
        nombre_partants=course.nombre_partants,
        dotation=course.dotation,
        categorie=course.categorie,
        statut=course.statut,
        partants=partant_schemas,
        predictions=pred_schemas,
    )


@router.get("/previsions/jour", response_model=list[PrevisionJourSchema])
async def previsions_jour(
    db: AsyncSession = Depends(get_db),
):
    """Previsions du jour triees par score de confiance decroissant, avec cote."""
    today = date.today()
    tomorrow = today + timedelta(days=1)

    # Courses du jour (A_VENIR ou TERMINE si la course vient juste de passer)
    stmt = (
        select(Course)
        .where(
            Course.date >= datetime.combine(today, datetime.min.time()),
            Course.date < datetime.combine(tomorrow, datetime.min.time()),
        )
    )
    result = await db.execute(stmt)
    courses = result.scalars().all()

    items = []
    for course in courses:
        # Prediction rang 1 (favori du modele)
        pred_stmt = (
            select(Prediction)
            .where(Prediction.course_id == course.id, Prediction.rang_predit == 1)
        )
        pred = (await db.execute(pred_stmt)).scalar_one_or_none()
        if not pred:
            continue

        partant = (await db.execute(
            select(Partant).where(Partant.id == pred.partant_id)
        )).scalar_one_or_none()
        if not partant:
            continue

        cheval = (await db.execute(
            select(Cheval).where(Cheval.id == partant.cheval_id)
        )).scalar_one_or_none()

        jockey_nom = None
        if partant.jockey_id:
            jockey = (await db.execute(
                select(Jockey).where(Jockey.id == partant.jockey_id)
            )).scalar_one_or_none()
            jockey_nom = jockey.nom if jockey else None

        entraineur_nom = None
        if partant.entraineur_id:
            ent = (await db.execute(
                select(Entraineur).where(Entraineur.id == partant.entraineur_id)
            )).scalar_one_or_none()
            entraineur_nom = ent.nom if ent else None

        items.append(PrevisionJourSchema(
            course_id=course.id,
            hippodrome=course.hippodrome,
            numero_reunion=course.numero_reunion,
            numero_course=course.numero_course,
            heure=course.date.strftime("%H:%M"),
            distance=course.distance,
            nombre_partants=course.nombre_partants,
            cheval_nom=cheval.nom if cheval else "?",
            numero=partant.numero,
            jockey_nom=jockey_nom,
            entraineur_nom=entraineur_nom,
            cote=partant.cote_depart or partant.cote_probable,
            probabilite=pred.probabilite,
            score_confiance=pred.score_confiance,
            is_value_bet=pred.is_value_bet,
            top5_confiance=pred.top5_confiance,
            commentaire=pred.commentaire,
        ))

    # Tri par confiance decroissante
    items.sort(key=lambda x: x.score_confiance, reverse=True)
    return items


@router.get("/backtesting/stats", response_model=BacktestingStatsSchema)
async def backtesting_stats(
    mise: float = Query(1.0, description="Mise unitaire en €"),
    db: AsyncSession = Depends(get_db),
):
    """Statistiques globales de backtesting sur toutes les courses terminées."""
    # Total courses terminées
    total_stmt = select(func.count(Course.id)).where(Course.statut == "TERMINE")
    total_result = await db.execute(total_stmt)
    total_courses = total_result.scalar() or 0

    # Courses avec prédiction rang 1
    pred_stmt = (
        select(Prediction)
        .where(Prediction.rang_predit == 1, Prediction.resultat_gagnant.is_not(None))
    )
    pred_result = await db.execute(pred_stmt)
    preds = pred_result.scalars().all()

    courses_predites = len(preds)
    gagnant_correct = sum(1 for p in preds if p.resultat_gagnant)
    place_correct = sum(1 for p in preds if p.resultat_place)

    profit_gagnant = sum(
        (p.gain_gagnant or -mise) for p in preds
    )
    profit_place = sum(
        (p.gain_place or -mise) for p in preds
    )

    total_mise = courses_predites * mise if courses_predites else 1

    return BacktestingStatsSchema(
        total_courses=total_courses,
        courses_predites=courses_predites,
        gagnant_correct=gagnant_correct,
        place_correct=place_correct,
        taux_gagnant=round(gagnant_correct / max(courses_predites, 1) * 100, 1),
        taux_place=round(place_correct / max(courses_predites, 1) * 100, 1),
        roi_gagnant=round(profit_gagnant / total_mise * 100, 1),
        roi_place=round(profit_place / total_mise * 100, 1),
        profit_gagnant=round(profit_gagnant, 2),
        profit_place=round(profit_place, 2),
        mise_unitaire=mise,
    )


@router.get("/bilan-veille", response_model=BilanVeilleSummarySchema)
async def bilan_veille(db: AsyncSession = Depends(get_db)):
    """Bilan des 5 confiances figées de la veille (top5_confiance=True) :
    résultat et gains simulés avec 1€ sur le gagnant et 4€ sur le placé."""
    yesterday = date.today() - timedelta(days=1)
    yesterday_start = datetime.combine(yesterday, datetime.min.time())
    today_start = datetime.combine(date.today(), datetime.min.time())

    # Courses de la veille terminées
    stmt = (
        select(Course)
        .where(
            Course.date >= yesterday_start,
            Course.date < today_start,
            Course.statut == "TERMINE",
        )
    )
    result = await db.execute(stmt)
    courses = result.scalars().all()
    course_ids = [c.id for c in courses]
    course_map = {c.id: c for c in courses}

    if not course_ids:
        return BilanVeilleSummarySchema(
            date_veille=yesterday.strftime("%d/%m/%Y"),
            bilans=[], total_mise=0, total_gain=0, profit=0,
        )

    # Récupérer uniquement les prédictions marquées top5_confiance
    pred_stmt = (
        select(Prediction)
        .where(
            Prediction.course_id.in_(course_ids),
            Prediction.top5_confiance == True,
        )
        .order_by(Prediction.score_confiance.desc())
    )
    preds = (await db.execute(pred_stmt)).scalars().all()

    items = []
    for pred in preds:
        course = course_map.get(pred.course_id)
        if not course:
            continue

        partant = (await db.execute(
            select(Partant).where(Partant.id == pred.partant_id)
        )).scalar_one_or_none()
        if not partant:
            continue

        cheval = (await db.execute(
            select(Cheval).where(Cheval.id == partant.cheval_id)
        )).scalar_one_or_none()

        if pred.resultat_gagnant:
            resultat = "gagnant"
        elif pred.resultat_place:
            resultat = "place"
        else:
            resultat = "perdu"

        gain_g = (partant.rapport_gagnant or 0) - 1.0 if pred.resultat_gagnant else -1.0
        gain_p = ((partant.rapport_place or 0) * 4) - 4.0 if pred.resultat_place else -4.0

        items.append(BilanVeilleSchema(
            course_id=course.id,
            hippodrome=course.hippodrome,
            numero_reunion=course.numero_reunion,
            numero_course=course.numero_course,
            heure=course.date.strftime("%H:%M"),
            cheval_nom=cheval.nom if cheval else "?",
            numero=partant.numero,
            cote=partant.cote_depart or partant.cote_probable,
            score_confiance=pred.score_confiance,
            resultat=resultat,
            classement=partant.classement,
            gain_gagnant=round(gain_g, 2),
            gain_place=round(gain_p, 2),
        ))

    total_gain = sum(b.gain_gagnant + b.gain_place for b in items)
    total_mise = len(items) * 5.0

    return BilanVeilleSummarySchema(
        date_veille=yesterday.strftime("%d/%m/%Y"),
        bilans=items,
        total_mise=total_mise,
        total_gain=round(total_mise + total_gain, 2),
        profit=round(total_gain, 2),
    )


@router.get("/confiance/stats", response_model=ConfianceStatsSchema)
async def confiance_stats(db: AsyncSession = Depends(get_db)):
    """Stats historiques basées sur le flag top5_confiance figé en base.
    Taux gagnant/placé sur toute la profondeur d'historique."""
    # Toutes les prédictions marquées top5_confiance avec résultat backtesté
    pred_stmt = (
        select(Prediction)
        .where(
            Prediction.top5_confiance == True,
            Prediction.resultat_gagnant.is_not(None),
        )
    )
    preds = (await db.execute(pred_stmt)).scalars().all()

    total = len(preds)
    gagnant_ok = 0
    place_ok = 0
    profit_g = 0.0
    profit_p = 0.0

    for pred in preds:
        partant = (await db.execute(
            select(Partant).where(Partant.id == pred.partant_id)
        )).scalar_one_or_none()

        if pred.resultat_gagnant:
            gagnant_ok += 1
            profit_g += (partant.rapport_gagnant or 0) - 1.0 if partant else -1.0
        else:
            profit_g -= 1.0
        if pred.resultat_place:
            place_ok += 1
            profit_p += ((partant.rapport_place or 0) * 4) - 4.0 if partant else -4.0
        else:
            profit_p -= 4.0

    perdu = total - place_ok

    return ConfianceStatsSchema(
        total_courses=total,
        gagnant_correct=gagnant_ok,
        place_correct=place_ok,
        perdu=perdu,
        taux_gagnant=round(gagnant_ok / max(total, 1) * 100, 1),
        taux_place=round(place_ok / max(total, 1) * 100, 1),
        profit_gagnant_1e=round(profit_g, 2),
        profit_place_4e=round(profit_p, 2),
        profit_total=round(profit_g + profit_p, 2),
    )
