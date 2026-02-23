from datetime import datetime
from sqlalchemy import String, Integer, Float, ForeignKey, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"), index=True)
    partant_id: Mapped[int] = mapped_column(ForeignKey("partants.id"), index=True)

    probabilite: Mapped[float] = mapped_column(Float)  # 0.0–1.0, probabilité prédite
    rang_predit: Mapped[int] = mapped_column(Integer)  # 1=favori modèle
    score_confiance: Mapped[float] = mapped_column(Float)  # 0–100, confiance du pari
    is_value_bet: Mapped[bool] = mapped_column(default=False)
    top5_confiance: Mapped[bool] = mapped_column(default=False)  # True si dans le top 5 confiance du jour
    commentaire: Mapped[str | None] = mapped_column(Text)  # justification textuelle

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Backtesting (rempli après la course)
    resultat_gagnant: Mapped[bool | None] = mapped_column(default=None)  # True si classement == 1
    resultat_place: Mapped[bool | None] = mapped_column(default=None)  # True si classement <= 3
    gain_gagnant: Mapped[float | None] = mapped_column(Float)  # rapport * mise - mise (si gagnant)
    gain_place: Mapped[float | None] = mapped_column(Float)  # rapport * mise - mise (si placé)

    course: Mapped["Course"] = relationship(back_populates="predictions")
    partant: Mapped["Partant"] = relationship()
