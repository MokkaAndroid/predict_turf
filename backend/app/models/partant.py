from sqlalchemy import String, Integer, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Partant(Base):
    __tablename__ = "partants"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id"), index=True)
    cheval_id: Mapped[int] = mapped_column(ForeignKey("chevaux.id"), index=True)
    jockey_id: Mapped[int | None] = mapped_column(ForeignKey("jockeys.id"))
    entraineur_id: Mapped[int | None] = mapped_column(ForeignKey("entraineurs.id"))

    numero: Mapped[int] = mapped_column(Integer)
    poids: Mapped[float | None] = mapped_column(Float)  # en kg
    cote_probable: Mapped[float | None] = mapped_column(Float)
    cote_depart: Mapped[float | None] = mapped_column(Float)

    # Résultat
    classement: Mapped[int | None] = mapped_column(Integer)  # 1=gagnant, 2=deuxième, etc. 0=non classé
    temps: Mapped[str | None] = mapped_column(String(20))
    rapport_gagnant: Mapped[float | None] = mapped_column(Float)  # rapport pour 1€ en gagnant
    rapport_place: Mapped[float | None] = mapped_column(Float)  # rapport pour 1€ en placé
    statut: Mapped[str] = mapped_column(String(20), default="PARTANT")  # PARTANT, NON_PARTANT, DISQUALIFIE

    course: Mapped["Course"] = relationship(back_populates="partants")
    cheval: Mapped["Cheval"] = relationship(back_populates="participations")
    jockey: Mapped["Jockey"] = relationship(back_populates="participations")
    entraineur: Mapped["Entraineur"] = relationship(back_populates="participations")
