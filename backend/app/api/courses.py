from datetime import datetime, date, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models import Course, Partant, Prediction, Cheval, Jockey, Entraineur
from app.schemas.course import (
    CourseListSchema,
    CourseDetailSchema,
    PartantSchema,
    PredictionSchema,
    PrevisionJourSchema,
    BacktestingStatsSchema,
)

router = APIRouter(prefix="/api", tags=["courses"])


@router.get("/courses/passees", response_model=list[CourseListSchema])
async def courses_passees(
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    hippodrome: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Course)
        .where(Course.statut == "TERMINE")
        .order_by(Course.date.desc())
    )
    if hippodrome:
        stmt = stmt.where(Course.hippodrome.ilike(f"%{hippodrome}%"))
    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    courses = result.scalars().all()

    items = []
    for c in courses:
        # Chercher la prédiction rang 1 pour cette course
        pred_stmt = (
            select(Prediction)
            .where(Prediction.course_id == c.id, Prediction.rang_predit == 1)
        )
        pred_result = await db.execute(pred_stmt)
        pred = pred_result.scalar_one_or_none()

        favori_nom = None
        favori_confiance = None
        pred_gagnant = None
        pred_place = None
        gain_g = None
        gain_p = None

        if pred:
            # Récupérer le nom du cheval
            partant_stmt = select(Partant).where(Partant.id == pred.partant_id)
            pr = await db.execute(partant_stmt)
            partant = pr.scalar_one_or_none()
            if partant:
                cheval_stmt = select(Cheval).where(Cheval.id == partant.cheval_id)
                cr = await db.execute(cheval_stmt)
                cheval = cr.scalar_one_or_none()
                favori_nom = cheval.nom if cheval else None
            favori_confiance = pred.score_confiance
            pred_gagnant = pred.resultat_gagnant
            pred_place = pred.resultat_place
            gain_g = pred.gain_gagnant
            gain_p = pred.gain_place

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
            favori_nom=favori_nom,
            favori_confiance=favori_confiance,
            prediction_correcte_gagnant=pred_gagnant,
            prediction_correcte_place=pred_place,
            gain_simule_gagnant=gain_g,
            gain_simule_place=gain_p,
        ))
    return items


@router.get("/courses/a-venir", response_model=list[CourseListSchema])
async def courses_a_venir(
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Course)
        .where(Course.statut == "A_VENIR")
        .order_by(Course.date.asc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    courses = result.scalars().all()

    items = []
    for c in courses:
        pred_stmt = (
            select(Prediction)
            .where(Prediction.course_id == c.id, Prediction.rang_predit == 1)
        )
        pred_result = await db.execute(pred_stmt)
        pred = pred_result.scalar_one_or_none()

        favori_nom = None
        favori_confiance = None
        if pred:
            partant_stmt = select(Partant).where(Partant.id == pred.partant_id)
            pr = await db.execute(partant_stmt)
            partant = pr.scalar_one_or_none()
            if partant:
                cheval_stmt = select(Cheval).where(Cheval.id == partant.cheval_id)
                cr = await db.execute(cheval_stmt)
                cheval = cr.scalar_one_or_none()
                favori_nom = cheval.nom if cheval else None
            favori_confiance = pred.score_confiance

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
            favori_nom=favori_nom,
            favori_confiance=favori_confiance,
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
