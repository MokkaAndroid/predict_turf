from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./hippique.db"
    redis_url: str = "redis://localhost:6379/0"
    pmu_base_url: str = "https://online.turfinfo.api.pmu.fr/rest/client/1"

    class Config:
        env_file = ".env"


settings = Settings()
