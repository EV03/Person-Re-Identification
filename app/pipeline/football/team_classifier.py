from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TeamColorObservation:
    team_id: str
    dominant_bgr: tuple[int, int, int]
    confidence: float


class SimpleTeamColorClassifier:
    """Small color-based placeholder for team assignment.

    It analyses the upper body region of a player crop and assigns the crop to
    the nearest configured team color. This is enough for first experiments, but
    not a final football analytics model.
    """

    def __init__(self, team_colors_bgr: dict[str, tuple[int, int, int]] | None = None) -> None:
        self.team_colors_bgr = team_colors_bgr or {
            "team_a": (255, 0, 0),
            "team_b": (0, 0, 255),
            "unknown": (128, 128, 128),
        }

    def classify(self, player_crop_bgr: np.ndarray) -> TeamColorObservation:
        if player_crop_bgr is None or player_crop_bgr.size == 0:
            return TeamColorObservation("unknown", (128, 128, 128), 0.0)

        height = player_crop_bgr.shape[0]
        upper_body = player_crop_bgr[: max(1, int(height * 0.55)), :, :]
        blurred = cv2.GaussianBlur(upper_body, (5, 5), 0)
        dominant = tuple(int(v) for v in np.mean(blurred.reshape(-1, 3), axis=0))

        best_team = "unknown"
        best_distance = float("inf")
        dominant_vec = np.asarray(dominant, dtype=np.float32)
        for team_id, color in self.team_colors_bgr.items():
            if team_id == "unknown":
                continue
            distance = float(np.linalg.norm(dominant_vec - np.asarray(color, dtype=np.float32)))
            if distance < best_distance:
                best_team = team_id
                best_distance = distance

        confidence = max(0.0, min(1.0, 1.0 - (best_distance / 442.0))) if best_distance != float("inf") else 0.0
        return TeamColorObservation(best_team, dominant, confidence)
