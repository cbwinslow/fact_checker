"""services/yolo_detector.py - Local YOLO object detection for frames.

Provides fast, local object detection using Ultralytics YOLOv8/11.
Runs on CPU or GPU, no API calls required.

Usage:
    detector = YOLODetector()
    detections = await detector.detect_frames(frame_paths)
    
Output: List of DetectionResult per frame with structured objects.

NOTE: This module handles missing/incompatible ultralytics gracefully.
If ultralytics is not installed or has compatibility issues, it will
log a warning and return empty detections (graceful degradation).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Classes (always available, no dependencies)
# ---------------------------------------------------------------------------

@dataclass
class DetectedObject:
    """Single object detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: List[float]      # [x1, y1, x2, y2] in pixels
    bbox_xywhn: List[float]     # [x_center, y_center, width, height] normalized 0-1


@dataclass
class FrameDetections:
    """All detections for a single frame."""
    frame_path: Path
    frame_index: int
    timestamp_sec: Optional[float] = None
    detections: List[DetectedObject] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0
    inference_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Lazy YOLO Detector (imports ultralytics only when instantiated)
# ---------------------------------------------------------------------------

class YOLODetector:
    """Local YOLO object detector for video frames.
    
    Handles missing/incompatible ultralytics gracefully by returning
    empty detections instead of crashing the pipeline.
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",  # nano for speed; yolov8s/m/l/x for accuracy
        device: str = "cpu",              # "cpu", "cuda", "mps"
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 100,
    ):
        self.model_name = model_name
        self.device = device
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self._model = None
        self._class_names = {}
        self._available = False
        
        # Try to load model lazily
        self._try_load_model()
    
    def _try_load_model(self):
        """Attempt to load YOLO model, handle failures gracefully."""
        try:
            from ultralytics import YOLO
            log.info("[YOLODetector] Loading model: {} on {}", 
                            self.model_name, self.device)
            self._model = YOLO(self.model_name)
            self._model.to(self.device)
            self._class_names = self._model.names if hasattr(self._model, 'names') else {}
            self._available = True
            log.info(f"[YOLODetector] Model loaded successfully")
        except Exception as exc:
            log.warning(
                "[YOLODetector] Failed to load YOLO model (will skip object detection): {}",
                exc
            )
            self._model = None
            self._available = False
    
    @property
    def available(self) -> bool:
        """Check if YOLO detector is available."""
        return self._available and self._model is not None
    
    async def detect_frames(
        self,
        frame_paths: List[Path],
        timestamps: Optional[List[float]] = None,
    ) -> List[FrameDetections]:
        """Run detection on multiple frames concurrently."""
        if not frame_paths:
            return []
        
        if not self.available:
            log.debug("[YOLODetector] Not available, returning empty detections")
            return [
                FrameDetections(
                    frame_path=p,
                    frame_index=i,
                    timestamp_sec=timestamps[i] if timestamps and i < len(timestamps) else None,
                )
                for i, p in enumerate(frame_paths)
            ]
        
        sem = asyncio.Semaphore(2)  # Limit concurrent inferences
        
        async def _detect_one(idx: int, frame_path: Path) -> FrameDetections:
            async with sem:
                return await self._detect_single_frame(
                    frame_path, 
                    idx, 
                    timestamps[idx] if timestamps and idx < len(timestamps) else None
                )
        
        tasks = [_detect_one(i, p) for i, p in enumerate(frame_paths)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions gracefully
        results_out = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.warning(f"[YOLODetector] Frame {i} detection failed: {result}")
                results_out.append(FrameDetections(
                    frame_path=frame_paths[i],
                    frame_index=i,
                    timestamp_sec=timestamps[i] if timestamps and i < len(timestamps) else None,
                ))
            else:
                results_out.append(result)
        return results_out
    
    async def _detect_single_frame(
        self,
        frame_path: Path,
        frame_index: int,
        timestamp_sec: Optional[float] = None,
    ) -> FrameDetections:
        """Run detection on a single frame."""
        import time
        start = time.perf_counter()
        
        # Run inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,
                lambda: self._model.predict(
                    source=str(frame_path),
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_det,
                    verbose=False,
                    device=self.device,
                )
            )
        except Exception as exc:
            log.warning(f"[YOLODetector] Inference failed for {frame_path}: {exc}")
            return FrameDetections(
                frame_path=frame_path,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
            )
        
        inference_time = (time.perf_counter() - start) * 1000
        
        detections: List[DetectedObject] = []
        img_width = 0
        img_height = 0
        
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                img_height, img_width = r.orig_shape[:2]
                
                for box in r.boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    xywhn = box.xywhn[0].tolist()  # normalized
                    
                    detections.append(DetectedObject(
                        class_id=cls_id,
                        class_name=self._class_names.get(cls_id, f"class_{cls_id}"),
                        confidence=conf,
                        bbox_xyxy=xyxy,
                        bbox_xywhn=xywhn,
                    ))
        
        return FrameDetections(
            frame_path=frame_path,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            detections=detections,
            image_width=img_width,
            image_height=img_height,
            inference_time_ms=inference_time,
        )
    
    def to_vision_llm_context(self, frame_detections: List[FrameDetections]) -> str:
        """Format detections as structured context for Vision LLM."""
        if not frame_detections:
            return "No object detections available."
        
        lines = ["OBJECT DETECTION RESULTS (YOLO):\n"]
        
        for fd in frame_detections:
            ts = f" @ {fd.timestamp_sec:.1f}s" if fd.timestamp_sec else ""
            lines.append(f"Frame {fd.frame_index}{ts} ({fd.image_width}x{fd.image_height}):")
            
            if fd.detections:
                # Group by class
                by_class: Dict[str, List[DetectedObject]] = {}
                for det in fd.detections:
                    by_class.setdefault(det.class_name, []).append(det)
                
                for class_name, dets in by_class.items():
                    count = len(dets)
                    avg_conf = sum(d.confidence for d in dets) / count
                    lines.append(f"  - {class_name}: {count} instances (avg conf: {avg_conf:.2f})")
                    
                    # Show top 3 highest confidence
                    for det in sorted(dets, key=lambda d: d.confidence, reverse=True)[:3]:
                        x1, y1, x2, y2 = det.bbox_xyxy
                        lines.append(
                            f"      • conf={det.confidence:.2f} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
                        )
            else:
                lines.append("  - No objects detected above threshold")
        
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory function (singleton)
# ---------------------------------------------------------------------------

_yolo_detector: Optional[YOLODetector] = None


def get_yolo_detector(
    model_name: str = "yolov8n.pt",
    device: str = "cpu",
) -> YOLODetector:
    """Get or create singleton YOLO detector.
    
    Returns a detector instance that gracefully handles missing/incompatible
    ultralytics by returning empty detections instead of crashing.
    """
    global _yolo_detector
    if _yolo_detector is None:
        _yolo_detector = YOLODetector(model_name=model_name, device=device)
    return _yolo_detector