from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BallObservation:
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    pitch_x: float | None = None
    pitch_y: float | None = None


class BallDetectorPlaceholder:
    """Placeholder for a future football ball detector.

    The current YOLO model is COCO-based and is not reliable for football balls
    in broadcast footage. The planned next step is a custom detector with at
    least these classes: player, goalkeeper, referee, ball.
    """

    def detect(self, frame_bgr: np.ndarray) -> list[BallObservation]:
        _ = frame_bgr
        return []
