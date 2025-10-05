from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    # Для локальной разработки
    DB_HOST: str = "localhost"
    DB_USER: str = "postgres"
    DB_NAME: str = "explore_planets"
    DB_PASSWORD: str = "password"
    DB_PORT: str = "5432"
    DB_SCHEMA: str = "public"
    
    # Для Render - используем DATABASE_URL если доступен
    DATABASE_URL: str = ""

    TESS_MODELS_PATH: str = "app/assets/models/TESS/"
    K2_MODEL_PATH: str = "app/assets/models/K2/"
    FEATURES_PATH: str = "app/assets/features/"

    @property
    def db_url(self) -> str:
        # Если есть DATABASE_URL (например, на Render), используем его
        if self.DATABASE_URL:
            # Преобразуем postgres:// в postgresql+asyncpg://
            return self.DATABASE_URL.replace("postgres://", "postgresql+asyncpg://")
        
        # Иначе используем отдельные параметры
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    class Config:
        env_file = ".env"


settings = Settings()