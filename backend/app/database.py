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

        # Migration : ajouter la colonne top5_confiance si elle n'existe pas
        from sqlalchemy import text, inspect
        def _check_column(sync_conn):
            insp = inspect(sync_conn)
            columns = [col["name"] for col in insp.get_columns("predictions")]
            return "top5_confiance" in columns

        has_col = await conn.run_sync(_check_column)
        if not has_col:
            await conn.execute(
                text("ALTER TABLE predictions ADD COLUMN top5_confiance BOOLEAN DEFAULT FALSE")
            )
