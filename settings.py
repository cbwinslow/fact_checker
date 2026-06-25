from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file.

    Per-task model overrides allow individual pipeline stages to be
    redirected to different models without touching code.  If an
    override is not set, the MODEL_REGISTRY defaults in openrouter.py
    are used.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # App
    # ------------------------------------------------------------------ #
    app_env: str = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ------------------------------------------------------------------ #
    # OpenRouter - core
    # ------------------------------------------------------------------ #
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="OPENROUTER_BASE_URL",
    )
    # Fallback model used when no task-specific override is set and the
    # task key is not found in MODEL_REGISTRY.
    openrouter_model: str = Field(
        default="openai/gpt-oss-120b:free",
        alias="OPENROUTER_MODEL",
    )

    # ------------------------------------------------------------------ #
    # OpenRouter - per-task model overrides (all optional)
    # Set these in .env to swap a specific pipeline stage to a different
    # model without changing any code.
    # ------------------------------------------------------------------ #
    model_extraction: str | None = Field(
        default=None,
        alias="MODEL_EXTRACTION",
        description="Override for claim extraction stage (default: openai/gpt-oss-120b:free)",
    )
    model_verification: str | None = Field(
        default=None,
        alias="MODEL_VERIFICATION",
        description="Override for fact verification stage (default: nvidia/nemotron-3-ultra:free)",
    )
    model_orchestration: str | None = Field(
        default=None,
        alias="MODEL_ORCHESTRATION",
        description="Override for pipeline orchestration stage (default: nvidia/nemotron-3-super:free)",
    )
    model_multimodal: str | None = Field(
        default=None,
        alias="MODEL_MULTIMODAL",
        description="Override for video/audio/image ingestion stage (default: nvidia/nemotron-3-nano-omni:free)",
    )
    model_tooluse: str | None = Field(
        default=None,
        alias="MODEL_TOOLUSE",
        description="Override for tool-use / DB-write stage (default: cohere/north-mini-code:free)",
    )
    model_fast: str | None = Field(
        default=None,
        alias="MODEL_FAST",
        description="Override for fast/lightweight subtasks (default: openai/gpt-oss-20b:free)",
    )

    # ------------------------------------------------------------------ #
    # Database
    # ------------------------------------------------------------------ #
    database_url: str = Field(default="", alias="DATABASE_URL")

    # ------------------------------------------------------------------ #
    # Whisper ASR
    # ------------------------------------------------------------------ #
    whisper_model_size: str = Field(default="base", alias="WHISPER_MODEL_SIZE")
    whisper_device: str = Field(default="cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field(default="int8", alias="WHISPER_COMPUTE_TYPE")

    # ------------------------------------------------------------------ #
    # Media / artifacts
    # ------------------------------------------------------------------ #
    artifact_dir: str = Field(default="./artifacts", alias="ARTIFACT_DIR")
    media_cache_dir: str = Field(default="./media_cache", alias="MEDIA_CACHE_DIR")

    # ------------------------------------------------------------------ #
    # FastAPI
    # ------------------------------------------------------------------ #
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # ------------------------------------------------------------------ #
    # ffmpeg
    # ------------------------------------------------------------------ #
    ffmpeg_bin: str = Field(default="ffmpeg", alias="FFMPEG_BIN")
    ffprobe_bin: str = Field(default="ffprobe", alias="FFPROBE_BIN")

    # ------------------------------------------------------------------ #
    # Optional external APIs
    # ------------------------------------------------------------------ #
    google_factcheck_api_key: str | None = Field(
        default=None, alias="GOOGLE_FACTCHECK_API_KEY"
    )
    serper_api_key: str | None = Field(default=None, alias="SERPER_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
