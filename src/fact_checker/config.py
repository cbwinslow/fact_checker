"""Centralised settings loaded from environment / .env file."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_model: str = "nvidia/nemotron-ultra-253b-v1:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Database
    database_url: str = "postgresql+asyncpg://fact_checker:password@localhost:5432/fact_checker"

    # Whisper ASR
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # App
    log_level: str = "INFO"
    artifact_dir: Path = Path("./artifacts")
    media_cache_dir: Path = Path("./media_cache")

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Evidence / Search
    google_factcheck_api_key: str = ""
    serper_api_key: str = ""


settings = Settings()
