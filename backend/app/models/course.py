from datetime import datetime
from sqlalchemy import String, Integer, Float, DateTime, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Course(Base):
    __tablename__ = "courses"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    pmu_id: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    date: Mapped[datetime] = mapped_column(DateTime, index=True)
    hippodrome: Mapped[str] = mapped_column(String(100))
    numero_reunion: Mapped[int] = mapped_column(Integer)
    numero_course: Mapped[int] = mapped_column(Integer)
    discipline: Mapped[str] = mapped_column(String(20), default="PLAT")
    surface: Mapped[str | None] = mapped_column(String(30))  # PSF, Herbe
    distance: Mapped[int | None] = mapped_column(Integer)  # en mètres
    condition_piste: Mapped[str | None] = mapped_column(String(30))  # Bon, Souple, Lourd
    nombre_partants: Mapped[int | None] = mapped_column(Integer)
    dotation: Mapped[float | None] = mapped_column(Float)
    categorie: Mapped[str | None] = mapped_column(String(100))  # ex: Quinté+, Listed, Groupe
    statut: Mapped[str] = mapped_column(String(20), default="A_VENIR")  # A_VENIR, TERMINE

    partants: Mapped[list["Partant"]] = relationship(back_populates="course", cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="course", cascade="all, delete-orphan")
