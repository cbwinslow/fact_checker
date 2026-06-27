"""Centralised settings loaded from environment / .env file.

When OPENROUTER_API_KEY is empty/unset (e.g. during local dev before the key
refreshes) build_chat_model() returns a MockChatModel that responds with
realistic stub JSON so the full pipeline can be exercised without hitting any
external API.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, List, Literal, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-task model registry - all free-tier models on OpenRouter
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
    # Structured claim extraction - best function calling + JSON output
    "extraction":    "openai/gpt-oss-120b:free",
    # Deep fact verification - 1M ctx, built for research & multi-step reasoning
    "verification":  "nvidia/nemotron-3-ultra-253b:free",
    # Pipeline orchestration - agent coherence & long-term planning
    "orchestration": "nvidia/nemotron-3-super-49b-v1:free",
    # Vision / image analysis - accepts image_url content parts (OpenAI vision format)
    # meta-llama/llama-4-maverick is free-tier and natively multimodal
    "multimodal":    "meta-llama/llama-4-maverick:free",
    # Tool use, structured DB writes, JSON schema output - lowest latency
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

    # Per-task model overrides (optional - set in .env to swap a stage)
    model_extraction:    Optional[str] = None
    model_verification:  Optional[str] = None
    model_orchestration: Optional[str] = None
    model_multimodal:    Optional[str] = None
    model_tooluse:       Optional[str] = None
    model_fast:          Optional[str] = None

    # Database
    database_url: str = "postgresql+asyncpg://fact_checker:password@localhost:5432/fact_checker"

    # Whisper ASR
    whisper_model_size:    str = "base"
    whisper_device:        str = "cpu"
    whisper_compute_type:  str = "int8"

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


# ---------------------------------------------------------------------------
# Mock LLM - used when OPENROUTER_API_KEY is unset
# ---------------------------------------------------------------------------

_MOCK_RESPONSES: dict[str, str] = {
    "extraction": json.dumps([
        {
            "text": "[MOCK] The Earth is flat.",
            "is_checkable": True,
            "confidence": 0.95,
            "context": "Mock claim for offline development",
        },
        {
            "text": "[MOCK] Vaccines cause autism.",
            "is_checkable": True,
            "confidence": 0.90,
            "context": "Mock claim for offline development",
        },
    ]),
    "verification": json.dumps({
        "verdict": "refuted",
        "confidence": 0.95,
        "explanation": "[MOCK] This claim is well-refuted by scientific consensus. "
                       "Running in offline/mock mode - no real verification performed.",
        "requires_human_review": False,
    }),
    # Mock vision analysis response for offline/no-key development
    "multimodal": json.dumps({
        "description": "[MOCK] A video frame showing a person speaking at a podium. "
                       "Running in offline/mock mode - no real image analysis performed.",
        "objects": [
            {"label": "person", "confidence": 0.95, "text_content": None},
            {"label": "podium", "confidence": 0.88, "text_content": None},
            {"label": "text_overlay", "confidence": 0.92,
             "text_content": "[MOCK] Breaking News: Earth is flat - Scientists say"},
        ],
        "text_in_image": "[MOCK] Breaking News: Earth is flat - Scientists say",
        "visible_claims": [
            "[MOCK] Earth is flat - visible as on-screen chyron",
        ],
        "context_notes": "[MOCK] Running in offline mode. No forensic analysis performed.",
        "manipulation_risk": "unknown",
        "manipulation_reason": "[MOCK] Cannot assess without live image analysis.",
    }),
    "default": json.dumps({"result": "[MOCK] No API key set - running in offline mode."}),
}


class MockChatModel(BaseChatModel):
    """Drop-in ChatModel stub for offline / no-API-key development.

    Returns deterministic JSON responses shaped to match what each pipeline
    agent expects, so the full pipeline can run end-to-end without any
    external API calls. Also handles the multimodal content-list format
    used by image_analyst.py (list of {type: text} + {type: image_url}).
    """
    task: str = "default"

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = _MOCK_RESPONSES.get(self.task, _MOCK_RESPONSES["default"])
        msg = AIMessage(content=content)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, run_manager, **kwargs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_chat_model(
    task: TaskType = "extraction",
    temperature: float = 0,
    max_tokens: Optional[int] = None,
) -> BaseChatModel:
    """Return a chat model pre-configured for the given pipeline task.

    Resolution order for the model ID:
      1. Per-task env override (e.g. MODEL_MULTIMODAL=...)
      2. MODEL_REGISTRY default for the task
      3. settings.openrouter_model global fallback

    If OPENROUTER_API_KEY is empty/unset, returns a MockChatModel that
    produces realistic stub responses so the pipeline can run fully offline.

    Note on vision models: for the "multimodal" task the returned ChatOpenAI
    instance is configured with the Llama-4-Maverick model which accepts
    image_url content parts in the OpenAI messages format. Ensure you are
    sending messages with content=[{type:text,...},{type:image_url,...}] 
    rather than a plain string when using this task slot.
    """
    s = get_settings()
    if not s.openrouter_api_key.strip():
        log.warning(
            "[config] OPENROUTER_API_KEY not set - using MockChatModel for task '%s'. "
            "Pipeline will run with stub responses.",
            task,
        )
        return MockChatModel(task=task)

    override  = getattr(s, f"model_{task}", None)
    model_id  = override or MODEL_REGISTRY.get(task, s.openrouter_model)

    kwargs: dict = dict(
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
        model=model_id,
        temperature=temperature,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return ChatOpenAI(**kwargs)
