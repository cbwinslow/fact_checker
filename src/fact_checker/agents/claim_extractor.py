"""ClaimExtractionAgent - extracts atomic checkable claims from transcript."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from uuid import UUID

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import settings
from ..models import Claim, TranscriptSegment

log = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "claim_extraction.md"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openrouter_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.1,
        max_tokens=4096,
    )


async def extract_claims(
    job_id: UUID,
    segments: list[TranscriptSegment],
    chunk_size: int = 50,
) -> list[Claim]:
    """Send transcript chunks to the LLM; return parsed Claim objects."""
    llm = _build_llm()
    system_prompt = _load_prompt()
    all_claims: list[Claim] = []

    # Build transcript text with timestamps
    transcript_lines = [
        f"[{s.start_sec:.1f}s] {s.text}"
        for s in segments
    ]

    # Chunk to avoid context overflow on very long videos
    for i in range(0, len(transcript_lines), chunk_size):
        chunk = "\n".join(transcript_lines[i:i + chunk_size])
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"TRANSCRIPT CHUNK:\n{chunk}\n\nExtract all checkable factual claims as JSON."),
        ]
        log.info("[ClaimExtractor] Processing chunk %d/%d", i // chunk_size + 1, -(-len(transcript_lines) // chunk_size))
        response = await llm.ainvoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        try:
            items = json.loads(raw)
            if isinstance(items, dict) and "claims" in items:
                items = items["claims"]
            for item in items:
                claim = Claim(
                    job_id=job_id,
                    text=item.get("claim", item.get("text", "")),
                    is_checkable=item.get("is_checkable", True),
                    confidence=float(item.get("confidence", 0.8)),
                    context=item.get("context"),
                )
                if claim.text:
                    all_claims.append(claim)
        except json.JSONDecodeError as e:
            log.warning("[ClaimExtractor] JSON parse error: %s | raw=%s", e, raw[:200])

    log.info("[ClaimExtractor] Extracted %d claims from %d segments", len(all_claims), len(segments))
    return all_claims
