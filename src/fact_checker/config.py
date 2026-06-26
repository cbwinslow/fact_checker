"""Centralised settings loaded from environment / .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Per-task model registry — all free-tier models on OpenRouter
# ---------------------------------------------------------------------------
TaskType = Literal[
    "extraction",
    "verification",
    "orchestration",
    "multimodal",
    "tooluse",
    "fast",
]

MODEL_REGISTRY: dict[str, str] = {
    # Structured claim extraction — best function calling + JSON output
    "extraction":    "openai/gpt-oss-120b:free",
    # Deep fact verification — 1M ctx, built for research & multi-step reasoning
    "verification":  "nvidia/nemotron-3-ultra-253b:free",
    # Pipeline orchestration — agent coherence & long-term planning
    "orchestration": "nvidia/nemotron-3-super-49b-v1:free",
    # Video / audio / image ingestion — only free model accepting AV input
    "multimodal":    "nvidia/llama-3.1-nemotron-nano-8b-v1:free",
    # Tool use, structured DB writes, JSON schema output — lowest latency
    "tooluse":       "cohere/north-mini-code:free",
    # Quick low-cost subtasks (title gen, short summaries, routing)
    "fast":          "openai/gpt-oss-20b:free",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter
    openrouter_api_key:  str = ""
    openrouter_model:    str = "openai/gpt-oss-120b:free"  # fallback default
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Per-task model overrides (optional — set in .env to swap a stage)
    model_extraction:    Optional[str] = None
    model_verification:  Optional[str] = None
    model_orchestration: Optional[str] = None
    model_multimodal:    Optional[str] = None
    model_tooluse:       Optional[str] = None
    model_fast:          Optional[str] = None

    # Database
    database_url: str = "postgresql+asyncpg://fact_checker:password@localhost:5432/fact_checker"

    # Whisper ASR
    whisper_model_size:   str = "base"
    whisper_device:       str = "cpu"
    whisper_compute_type: str = "int8"

    # App
    log_level:       str  = "INFO"
    artifact_dir:    Path = Path("./artifacts")
    media_cache_dir: Path = Path("./media_cache")

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Evidence / Search
    google_factcheck_api_key: str = ""
    serper_api_key:           str = ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# Module-level singleton (backwards-compat with `from .config import settings`)
settings = get_settings()


def build_chat_model(
    task: TaskType = "extraction",
    temperature: float = 0,
    max_tokens: Optional[int] = None,
) -> ChatOpenAI:
    """Return a ChatOpenAI client pre-configured for the given pipeline task.

    Resolution order for the model ID:
      1. Per-task env override  (e.g. MODEL_EXTRACTION=...)
      2. MODEL_REGISTRY default for the task
      3. settings.openrouter_model global fallback
    """
    s = get_settings()
    override = getattr(s, f"model_{task}", None)
    model_id = override or MODEL_REGISTRY.get(task, s.openrouter_model)

    kwargs: dict = dict(
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
        model=model_id,
        temperature=temperature,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return ChatOpenAI(**kwargs)
