from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Entraineur(Base):
    __tablename__ = "entraineurs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    nom: Mapped[str] = mapped_column(String(100), index=True)

    participations: Mapped[list["Partant"]] = relationship(back_populates="entraineur")
