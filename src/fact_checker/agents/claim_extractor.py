"""ClaimExtractionAgent - extracts atomic checkable claims from transcript."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import UUID

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import build_chat_model
from ..models import Claim, TranscriptSegment

log = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "claim_extraction.md"

_FALLBACK_PROMPT = """You are a fact-checking assistant. Extract all specific, verifiable factual claims from the transcript below.
Return a JSON array of objects with keys: text, is_checkable, confidence (0-1), context.
Only return the JSON array, no other text."""


def _load_prompt() -> str:
    """Load prompt from file, fall back to inline default if missing."""
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.warning("[claim_extractor] Prompt file not found at %s, using fallback", PROMPT_PATH)
        return _FALLBACK_PROMPT


def _build_llm() -> BaseChatModel:
    """Use extraction task slot -> openai/gpt-oss-120b:free for structured JSON output."""
    return build_chat_model(task="extraction", temperature=0.1, max_tokens=4096)


async def extract_claims(
    job_id: UUID,
    segments: list[TranscriptSegment],
) -> list[Claim]:
    """Extract checkable claims from transcript segments."""
    if not segments:
        return []

    llm = _build_llm()
    system_prompt = _load_prompt()
    full_text = "\n".join(
        f"[{s.start_sec:.1f}s - {s.end_sec:.1f}s] {s.text}" for s in segments
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"TRANSCRIPT:\n{full_text}"),
    ]

    try:
        response = llm.invoke(messages)
        # Strip markdown code fences if present
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        raw: list[dict] = json.loads(content)
    except Exception as exc:
        log.error("[claim_extractor] LLM call or JSON parse failed: %s", exc)
        return []

    claims: list[Claim] = []
    for item in raw:
        try:
            claims.append(
                Claim(
                    job_id=job_id,
                    segment_id=None,
                    text=item["text"],
                    is_checkable=item.get("is_checkable", True),
                    confidence=float(item.get("confidence", 1.0)),
                    context=item.get("context"),
                )
            )
        except Exception as exc:
            log.warning("[claim_extractor] Skipping malformed claim item: %s", exc)
    log.info("[claim_extractor] Extracted %d claims from %d segments", len(claims), len(segments))
    return claims
