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
    top5_confiance: bool = False
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
    top5_confiance: bool = False
    commentaire: str | None = None

    class Config:
        from_attributes = True


class BilanVeilleSchema(BaseModel):
    """Bilan d'une prédiction de la veille (top 5 confiance)."""
    course_id: int
    hippodrome: str
    numero_reunion: int
    numero_course: int
    heure: str
    cheval_nom: str
    numero: int
    cote: float | None = None
    score_confiance: float
    resultat: str  # "gagnant", "place", "perdu"
    classement: int | None = None
    gain_gagnant: float  # avec mise 1€
    gain_place: float  # avec mise 4€

    class Config:
        from_attributes = True


class BilanVeilleSummarySchema(BaseModel):
    date_veille: str
    bilans: list[BilanVeilleSchema] = []
    total_mise: float  # 5 courses × (1€ gagnant + 4€ placé) = 25€
    total_gain: float
    profit: float

    class Config:
        from_attributes = True


class ConfianceStatsSchema(BaseModel):
    """Stats historiques des top-5 confiance."""
    total_courses: int
    gagnant_correct: int
    place_correct: int
    perdu: int
    taux_gagnant: float
    taux_place: float
    profit_gagnant_1e: float
    profit_place_4e: float
    profit_total: float

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
