"""ImageAnalystAgent - vision analysis of images and video frames.

For each image / frame:
  1. Extract file metadata (EXIF, dimensions) via services/vision.py.
  2. Encode the image as a base64 data URL.
  3. Call a vision-capable LLM (multimodal task slot) with a structured prompt.
  4. Parse the JSON response into an ImageAnalysis domain object.
  5. Surface any visible text claims back to the claim extraction pipeline.

Offline / no-key mode: Falls back to MockChatModel which returns stub data
so the pipeline can be exercised end-to-end without an API key.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional
from uuid import UUID

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import build_chat_model
from ..models import (
    DetectedObject,
    ImageAnalysis,
    ImageMetadata,
    ImageSourceType,
    ManipulationRisk,
)
from ..services.vision import image_to_data_url, read_image_metadata
from ..services.yolo_detector import get_yolo_detector, YOLODetector, FrameDetections

log = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "image_analysis.md"

_FALLBACK_PROMPT = """You are a forensic fact-checking image analyst. Analyse the provided image and return ONLY a JSON object with these keys:

- description (str): 2-4 sentence scene description.
- objects (list): each item has {"label": str, "confidence": float 0-1, "text_content": str|null}.
- text_in_image (str): all visible text, OCR-style, verbatim.
- visible_claims (list[str]): any factual assertions visible in the image (e.g. on-screen graphics, captions, chyrons, headlines).
- context_notes (str): forensic notes - lighting inconsistencies, metadata-image date mismatches, AI-generation artefacts, editing watermarks, etc.
- manipulation_risk (str): one of low | medium | high | unknown.
- manipulation_reason (str): brief rationale for the manipulation risk score.

Return ONLY the JSON object, no markdown fences."""


def _load_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.warning("[image_analyst] Prompt file not found at %s, using fallback", PROMPT_PATH)
        return _FALLBACK_PROMPT


def _build_llm() -> BaseChatModel:
    """Use the multimodal task slot - routes to a vision-capable model."""
    return build_chat_model(task="multimodal", temperature=0.0, max_tokens=2048)


async def analyse_images(
    job_id: UUID,
    image_paths: List[Path | str],
    *,
    source_type: ImageSourceType = ImageSourceType.VIDEO_FRAME,
    frame_timestamps: Optional[List[float]] = None,
) -> List[ImageAnalysis]:
    """Analyse a list of image files and return ImageAnalysis results.

    Args:
        job_id:          The UUID of the parent VideoJob.
        image_paths:     Paths to the image files to analyse.
        source_type:     How the images were obtained (default: VIDEO_FRAME).
        frame_timestamps: Optional list of timestamps (sec) corresponding to
                          each image in image_paths (for video frames).

    Returns:
        List of ImageAnalysis objects, one per input image.
    """
    if not image_paths:
        return []

    llm = _build_llm()
    system_prompt = _load_prompt()
    results: List[ImageAnalysis] = []

    for idx, raw_path in enumerate(image_paths):
        image_path = Path(raw_path)
        frame_sec = (
            frame_timestamps[idx]
            if frame_timestamps and idx < len(frame_timestamps)
            else None
        )

        # --- Step 1: File metadata (EXIF etc.) ---
        metadata = read_image_metadata(image_path)
        if frame_sec is not None:
            metadata.frame_timestamp_sec = frame_sec

        # --- Step 2: Local YOLO object detection (optional, fast, local) ---
        yolo_context = ""
        try:
            yolo_detector = get_yolo_detector()
            frame_detections = await yolo_detector.detect_frames(
                [image_path],
                [frame_sec] if frame_sec is not None else None,
            )
            if frame_detections:
                yolo_context = yolo_detector.to_vision_llm_context(frame_detections)
                log.debug(f"[image_analyst] YOLO detected objects for {image_path.name}")
        except Exception as exc:
            log.warning(f"[image_analyst] YOLO detection failed (continuing without): {exc}")

        # --- Step 3: Encode image as data URL ---
        try:
            data_url = image_to_data_url(image_path, max_side=1024)
        except Exception as exc:
            log.error(
                "[image_analyst] Failed to encode image %s: %s",
                image_path.name, exc,
            )
            results.append(_make_error_result(job_id, image_path, source_type, metadata, frame_sec, str(exc)))
            continue

        # --- Step 3: Call vision LLM ---
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            f"Image file: {image_path.name}\n"
                            f"Dimensions: {metadata.width}x{metadata.height}\n"
                            f"Format: {metadata.format}\n"
                            + (f"Camera: {metadata.camera_make} {metadata.camera_model}\n" if metadata.camera_make else "")
                            + (f"Date taken: {metadata.datetime_original}\n" if metadata.datetime_original else "")
                            + (f"Software: {metadata.software}\n" if metadata.software else "")
                            + (f"Frame at: {frame_sec:.1f}s\n" if frame_sec is not None else "")
                            + (f"\n{yolo_context}\n" if yolo_context else "")
                            + "\nPlease analyse this image:"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ]
            ),
        ]

        try:
            response = llm.invoke(messages)
            content = response.content
            if isinstance(content, str):
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                data: dict = json.loads(content)
            else:
                # Some LLMs return a list of content parts
                text_parts = [
                    p["text"] if isinstance(p, dict) else str(p)
                    for p in (content if isinstance(content, list) else [content])
                    if not (isinstance(p, dict) and p.get("type") == "image_url")
                ]
                data = json.loads(" ".join(text_parts).strip())

            objects = [
                DetectedObject(
                    label=obj.get("label", "unknown"),
                    confidence=float(obj.get("confidence", 0.0)),
                    bounding_box=obj.get("bounding_box"),
                    text_content=obj.get("text_content"),
                )
                for obj in data.get("objects", [])
            ]

            analysis = ImageAnalysis(
                job_id=job_id,
                source_type=source_type,
                source_path=str(image_path),
                frame_sec=frame_sec,
                metadata=metadata,
                description=data.get("description", ""),
                objects=objects,
                text_in_image=data.get("text_in_image", ""),
                visible_claims=data.get("visible_claims", []),
                context_notes=data.get("context_notes", ""),
                manipulation_risk=ManipulationRisk(
                    data.get("manipulation_risk", ManipulationRisk.UNKNOWN)
                ),
                manipulation_reason=data.get("manipulation_reason", ""),
            )

        except Exception as exc:
            log.error(
                "[image_analyst] LLM call or parse failed for %s: %s",
                image_path.name, exc,
            )
            analysis = _make_error_result(
                job_id, image_path, source_type, metadata, frame_sec, str(exc)
            )

        results.append(analysis)
        log.debug(
            "[image_analyst] Analysed %s: %s (%d objects, %d visible claims)",
            image_path.name,
            analysis.manipulation_risk.value,
            len(analysis.objects),
            len(analysis.visible_claims),
        )

    log.info(
        "[image_analyst] Completed analysis of %d/%d images for job %s",
        len(results), len(image_paths), job_id,
    )
    return results


def _make_error_result(
    job_id: UUID,
    image_path: Path,
    source_type: ImageSourceType,
    metadata: ImageMetadata,
    frame_sec: Optional[float],
    error_msg: str,
) -> ImageAnalysis:
    """Return a stub ImageAnalysis with error context when analysis fails."""
    return ImageAnalysis(
        job_id=job_id,
        source_type=source_type,
        source_path=str(image_path),
        frame_sec=frame_sec,
        metadata=metadata,
        description=f"Analysis failed: {error_msg}",
        manipulation_risk=ManipulationRisk.UNKNOWN,
        context_notes=f"[ERROR] {error_msg}",
    )
