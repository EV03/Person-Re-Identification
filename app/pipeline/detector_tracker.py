from __future__ import annotations

from typing import Iterable

import numpy as np

from app.storage.models import Detection


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
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("Ultralytics is not installed. Run: pip install -r requirements.txt") from exc

        self.model = YOLO(model_name)
        self.tracker = tracker
        self.confidence = confidence
        self.image_size = image_size
        self.device = None if device == "auto" else device
        self._fallback_track_id = 1

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

            if box.id is None:
                track_id = self._fallback_track_id
                self._fallback_track_id += 1
            else:
                track_id = int(box.id[0].detach().cpu().item())

            detections.append(
                Detection(
                    track_id=track_id,
                    bbox_xyxy=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                    confidence=conf,
                    class_id=class_id,
                )
            )

        return detections
