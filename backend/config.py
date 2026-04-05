from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    openai_api_key: str
    primary_model: str = "gpt-4o-mini"
    strong_model: str = "gpt-4o"
    secret_key: str
    database_url: str = "sqlite:///./bond.db"

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()