from typing import List

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./crazyitinerary.db"
    llm_api_url: AnyHttpUrl | None = None
    llm_api_key: str | None = None
    geoapify_api_key: str | None = None
    amadeus_api_key: str | None = None
    amadeus_api_secret: str | None = None
    frontend_origins: List[str] = ["http://localhost:3000"]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
