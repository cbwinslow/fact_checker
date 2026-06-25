from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI

from fact_checker.config.settings import get_settings

# ---------------------------------------------------------------------------
# Model registry - all free-tier models on OpenRouter, assigned by task type
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
    # Structured claim extraction - best function calling + structured output
    "extraction": "openai/gpt-oss-120b:free",
    # Deep fact verification - 1M ctx, built for research & multi-step reasoning
    "verification": "nvidia/nemotron-3-ultra-550b-a55b:free",
    # Pipeline orchestration - agent coherence & long-term planning
    "orchestration": "nvidia/nemotron-3-super-49b-v1:free",
    # Video / audio / image ingestion - only free model accepting AV input
    "multimodal": "nvidia/llama-3.1-nemotron-nano-8b-v1:free",
    # Tool use, structured DB writes, JSON schema output - lowest latency
    "tooluse": "cohere/north-mini-code:free",
    # Quick low-cost subtasks (title gen, short summaries, routing)
    "fast": "openai/gpt-oss-20b:free",
}


def build_chat_model(
    task: str = "extraction",
    temperature: float = 0,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """Return a ChatOpenAI client pre-configured for the given pipeline task.

    Args:
        task: One of 'extraction', 'verification', 'orchestration',
              'multimodal', 'tooluse', or 'fast'.  Falls back to the
              value of settings.openrouter_model if task is not found.
        temperature: Sampling temperature (default 0 for determinism).
        max_tokens: Optional cap on output tokens.

    Returns:
        A configured ChatOpenAI instance pointed at OpenRouter.
    """
    settings = get_settings()
    # Fall back to the env-configured model if task key is not found
    model_id = MODEL_REGISTRY.get(task, settings.openrouter_model)

    kwargs: dict = dict(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=model_id,
        temperature=temperature,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    return ChatOpenAI(**kwargs)
