from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import numpy as np

from app.storage.models import Detection
from app.config import PROJECT_ROOT, PipelineConfig
from app.evaluation.artifacts import file_reference, tracker_reference
from app.pipeline.contracts import PersonTracker


class UltralyticsPersonTracker:
    """Person detector + tracker using Ultralytics YOLO.

    The class uses YOLO's tracking mode on individual frames with persist=True.
    Only class 0 (person in COCO models) is returned.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        tracker: str = "bytetrack.yaml",
        confidence: float = 0.35,
        image_size: int = 640,
        device: str = "auto",
    ) -> None:
        if not 0 < confidence <= 1:
            raise ValueError("Tracking detection confidence must be positive and at most 1; Ultralytics replaces zero with 0.1.")
        # Keep library settings local and avoid network package installation
        # midway through an experiment. Respect explicit user overrides.
        if "YOLO_CONFIG_DIR" not in os.environ:
            config_dir = PROJECT_ROOT / "data" / "cache" / "ultralytics"
            config_dir.mkdir(parents=True, exist_ok=True)
            os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(f"Ultralytics backend could not be imported ({type(exc).__name__}: {exc}). Install requirements.txt.") from exc

        self.model = YOLO(model_name)
        self.tracker = tracker
        self.confidence = confidence
        self.image_size = image_size
        self.device = None if device == "auto" else device

    def describe_backend(self) -> dict[str, Any]:
        """Expose model provenance without leaking YOLO internals to the pipeline."""
        checkpoint = getattr(self.model, "ckpt_path", None)
        metadata: dict[str, Any] = {"tracker_config": self.tracker, "tracker": tracker_reference(self.tracker)}
        if checkpoint:
            metadata["detector"] = file_reference(Path(checkpoint))
        return metadata

    def track_frame(self, frame_bgr: np.ndarray) -> list[Detection]:
        results = self.model.track(
            frame_bgr,
            persist=True,
            tracker=self.tracker,
            classes=[0],
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
            conf = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else 0

            # A detector output without a tracker ID is not a stable track.
            # Never invent an ID in the same namespace as ByteTrack/BoT-SORT.
            track_id = None if box.id is None else int(box.id[0].detach().cpu().item())

            detections.append(
                Detection(
                    track_id=track_id,
                    bbox_xyxy=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                    confidence=conf,
                    class_id=class_id,
                )
            )

        return detections


def build_tracker(config: PipelineConfig) -> PersonTracker:
    """Default composition only; alternative factories need not use Ultralytics."""
    return UltralyticsPersonTracker(
        model_name=config.yolo_model, tracker=config.tracker,
        confidence=config.detection_confidence, image_size=config.image_size,
        device=config.device,
    )
