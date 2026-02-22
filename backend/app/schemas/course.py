from datetime import datetime
from pydantic import BaseModel


class PartantSchema(BaseModel):
    id: int
    numero: int
    cheval_nom: str
    jockey_nom: str | None = None
    entraineur_nom: str | None = None
    poids: float | None = None
    cote_probable: float | None = None
    cote_depart: float | None = None
    classement: int | None = None
    rapport_gagnant: float | None = None
    rapport_place: float | None = None
    statut: str

    class Config:
        from_attributes = True


class PredictionSchema(BaseModel):
    partant_id: int
    cheval_nom: str
    numero: int
    probabilite: float
    rang_predit: int
    score_confiance: float
    is_value_bet: bool
    commentaire: str | None = None

    class Config:
        from_attributes = True


class CourseListSchema(BaseModel):
    id: int
    pmu_id: str
    date: datetime
    hippodrome: str
    numero_reunion: int
    numero_course: int
    discipline: str
    distance: int | None = None
    nombre_partants: int | None = None
    statut: str
    # Résumé prédiction
    favori_nom: str | None = None
    favori_confiance: float | None = None
    # Backtesting
    prediction_correcte_gagnant: bool | None = None
    prediction_correcte_place: bool | None = None
    gain_simule_gagnant: float | None = None
    gain_simule_place: float | None = None

    class Config:
        from_attributes = True


class CourseDetailSchema(BaseModel):
    id: int
    pmu_id: str
    date: datetime
    hippodrome: str
    numero_reunion: int
    numero_course: int
    discipline: str
    surface: str | None = None
    distance: int | None = None
    condition_piste: str | None = None
    nombre_partants: int | None = None
    dotation: float | None = None
    categorie: str | None = None
    statut: str
    partants: list[PartantSchema] = []
    predictions: list[PredictionSchema] = []

    class Config:
        from_attributes = True


class PrevisionJourSchema(BaseModel):
    """Une prevision du jour : 1 ligne = 1 favori pour 1 course."""
    course_id: int
    hippodrome: str
    numero_reunion: int
    numero_course: int
    heure: str
    distance: int | None = None
    nombre_partants: int | None = None
    cheval_nom: str
    numero: int
    jockey_nom: str | None = None
    entraineur_nom: str | None = None
    cote: float | None = None
    probabilite: float
    score_confiance: float
    is_value_bet: bool
    commentaire: str | None = None

    class Config:
        from_attributes = True


class BacktestingStatsSchema(BaseModel):
    total_courses: int
    courses_predites: int
    gagnant_correct: int
    place_correct: int
    taux_gagnant: float  # %
    taux_place: float  # %
    roi_gagnant: float  # %
    roi_place: float  # %
    profit_gagnant: float  # en €
    profit_place: float  # en €
    mise_unitaire: float  # en €

    class Config:
        from_attributes = True
