from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PitchPoint:
    x_m: float
    y_m: float


class PitchMapper:
    """Image-to-pitch coordinate mapper based on a homography matrix.

    A later calibration step should estimate the matrix from detected pitch
    keypoints or manually selected field points. Until a matrix is set, mapping
    returns None so the stats layer does not treat pixel coordinates as meters.
    """

    def __init__(self, homography: np.ndarray | None = None) -> None:
        self.homography = homography

    def set_homography(self, image_points: np.ndarray, pitch_points: np.ndarray) -> None:
        if len(image_points) < 4 or len(pitch_points) < 4:
            raise ValueError("At least four image and pitch points are required for homography.")
        matrix, _ = cv2.findHomography(image_points.astype(np.float32), pitch_points.astype(np.float32))
        if matrix is None:
            raise ValueError("Could not compute pitch homography.")
        self.homography = matrix

    def map_image_point(self, x: float, y: float) -> PitchPoint | None:
        if self.homography is None:
            return None
        source = np.asarray([[[float(x), float(y)]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(source, self.homography)[0][0]
        return PitchPoint(x_m=float(mapped[0]), y_m=float(mapped[1]))

    @staticmethod
    def foot_point_from_bbox(bbox_xyxy: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, _y1, x2, y2 = bbox_xyxy
        return (float(x1 + x2) / 2.0, float(y2))
