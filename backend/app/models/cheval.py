from sqlalchemy import String, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Cheval(Base):
    __tablename__ = "chevaux"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    nom: Mapped[str] = mapped_column(String(100), index=True)
    age: Mapped[int | None] = mapped_column(Integer)
    sexe: Mapped[str | None] = mapped_column(String(10))  # M, F, H (hongre)
    race: Mapped[str | None] = mapped_column(String(100))
    pere: Mapped[str | None] = mapped_column(String(100))
    mere: Mapped[str | None] = mapped_column(String(100))
    proprietaire: Mapped[str | None] = mapped_column(String(200))

    participations: Mapped[list["Partant"]] = relationship(back_populates="cheval")
