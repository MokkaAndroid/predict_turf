from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(settings.async_database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with async_session() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        from sqlalchemy import text, inspect

        # Auto-migration: add missing columns to existing tables
        _migrations = [
            ("predictions", "top5_confiance", "BOOLEAN DEFAULT FALSE"),
            ("chevaux", "gains_carriere", "FLOAT"),
            ("partants", "rang_pronostic", "INTEGER"),
        ]

        def _get_existing_columns(sync_conn, table_name):
            insp = inspect(sync_conn)
            try:
                return [col["name"] for col in insp.get_columns(table_name)]
            except Exception:
                return []

        for table, column, col_type in _migrations:
            existing = await conn.run_sync(lambda sc, t=table: _get_existing_columns(sc, t))
            if column not in existing:
                await conn.execute(
                    text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                )
