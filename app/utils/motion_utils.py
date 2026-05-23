from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MotionSnapshot:
    """Per-frame motion information for one tracker id.

    The values are image-space values. They are intentionally used as a cheap
    support/debug signal for tracking and ReID decisions, not as a biometric
    feature. A later football mode can map these values to pitch coordinates.
    """

    track_id: int
    center_x: float
    center_y: float
    previous_center_x: float | None
    previous_center_y: float | None
    dx: float
    dy: float
    distance_px: float
    speed_px_per_sec: float
    direction_degrees: float | None
    direction_label: str
    frame_gap: int
    plausibility_score: float
    is_large_jump: bool

    def to_payload(self) -> dict[str, float | int | bool | str | None]:
        return {
            "track_id": self.track_id,
            "center_x": float(self.center_x),
            "center_y": float(self.center_y),
            "previous_center_x": None if self.previous_center_x is None else float(self.previous_center_x),
            "previous_center_y": None if self.previous_center_y is None else float(self.previous_center_y),
            "dx": float(self.dx),
            "dy": float(self.dy),
            "distance_px": float(self.distance_px),
            "speed_px_per_sec": float(self.speed_px_per_sec),
            "direction_degrees": None if self.direction_degrees is None else float(self.direction_degrees),
            "direction_label": self.direction_label,
            "frame_gap": int(self.frame_gap),
            "plausibility_score": float(self.plausibility_score),
            "is_large_jump": bool(self.is_large_jump),
        }


@dataclass
class _MotionState:
    center_x: float
    center_y: float
    frame_index: int
    smooth_dx: float = 0.0
    smooth_dy: float = 0.0


def bbox_center(bbox_xyxy: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def direction_label_from_vector(dx: float, dy: float, min_displacement_px: float = 2.0) -> str:
    if math.hypot(dx, dy) < min_displacement_px:
        return "still"

    # Image coordinates: +x right, +y down. For user-facing labels this is fine.
    abs_dx = abs(dx)
    abs_dy = abs(dy)

    if abs_dx >= abs_dy * 1.8:
        return "right" if dx > 0 else "left"
    if abs_dy >= abs_dx * 1.8:
        return "down" if dy > 0 else "up"

    vertical = "down" if dy > 0 else "up"
    horizontal = "right" if dx > 0 else "left"
    return f"{vertical}-{horizontal}"


def direction_degrees_from_vector(dx: float, dy: float, min_displacement_px: float = 2.0) -> float | None:
    if math.hypot(dx, dy) < min_displacement_px:
        return None
    # Convert image-space vector into a standard mathematical angle:
    # 0° = right, 90° = up, 180° = left, 270° = down.
    return float((math.degrees(math.atan2(-dy, dx)) + 360.0) % 360.0)


class MotionTracker:
    """Lightweight image-space movement tracker for existing ByteTrack ids.

    ByteTrack already performs the actual object tracking. This class only adds
    explainable movement metadata per track id: center displacement, speed,
    direction and jump plausibility.
    """

    def __init__(
        self,
        *,
        fps: float,
        frame_width: int,
        frame_height: int,
        max_jump_fraction: float = 0.20,
        smoothing_alpha: float = 0.35,
        min_displacement_px: float = 2.0,
    ) -> None:
        self.fps = max(float(fps), 1.0)
        self.frame_width = max(int(frame_width), 1)
        self.frame_height = max(int(frame_height), 1)
        self.frame_diagonal = math.hypot(float(self.frame_width), float(self.frame_height))
        self.max_jump_fraction = max(float(max_jump_fraction), 0.01)
        self.smoothing_alpha = min(max(float(smoothing_alpha), 0.0), 1.0)
        self.min_displacement_px = max(float(min_displacement_px), 0.0)
        self._states: dict[int, _MotionState] = {}

    def update(
        self,
        *,
        track_id: int,
        bbox_xyxy: tuple[int, int, int, int],
        frame_index: int,
    ) -> MotionSnapshot:
        center_x, center_y = bbox_center(bbox_xyxy)
        previous = self._states.get(track_id)

        if previous is None:
            self._states[track_id] = _MotionState(center_x=center_x, center_y=center_y, frame_index=frame_index)
            return MotionSnapshot(
                track_id=track_id,
                center_x=center_x,
                center_y=center_y,
                previous_center_x=None,
                previous_center_y=None,
                dx=0.0,
                dy=0.0,
                distance_px=0.0,
                speed_px_per_sec=0.0,
                direction_degrees=None,
                direction_label="new",
                frame_gap=0,
                plausibility_score=1.0,
                is_large_jump=False,
            )

        frame_gap = max(int(frame_index) - int(previous.frame_index), 1)
        raw_dx = center_x - previous.center_x
        raw_dy = center_y - previous.center_y

        if self.smoothing_alpha <= 0.0:
            smooth_dx = raw_dx
            smooth_dy = raw_dy
        else:
            smooth_dx = self.smoothing_alpha * raw_dx + (1.0 - self.smoothing_alpha) * previous.smooth_dx
            smooth_dy = self.smoothing_alpha * raw_dy + (1.0 - self.smoothing_alpha) * previous.smooth_dy

        distance_px = math.hypot(raw_dx, raw_dy)
        dt_seconds = frame_gap / self.fps
        speed_px_per_sec = distance_px / max(dt_seconds, 1e-6)

        allowed_distance = self.frame_diagonal * self.max_jump_fraction * max(1.0, math.sqrt(float(frame_gap)))
        jump_ratio = distance_px / max(allowed_distance, 1.0)
        plausibility_score = float(max(0.0, min(1.0, 1.0 - max(0.0, jump_ratio - 1.0))))
        is_large_jump = jump_ratio > 1.0

        direction_degrees = direction_degrees_from_vector(smooth_dx, smooth_dy, self.min_displacement_px)
        direction_label = direction_label_from_vector(smooth_dx, smooth_dy, self.min_displacement_px)

        self._states[track_id] = _MotionState(
            center_x=center_x,
            center_y=center_y,
            frame_index=frame_index,
            smooth_dx=smooth_dx,
            smooth_dy=smooth_dy,
        )

        return MotionSnapshot(
            track_id=track_id,
            center_x=center_x,
            center_y=center_y,
            previous_center_x=previous.center_x,
            previous_center_y=previous.center_y,
            dx=smooth_dx,
            dy=smooth_dy,
            distance_px=distance_px,
            speed_px_per_sec=speed_px_per_sec,
            direction_degrees=direction_degrees,
            direction_label=direction_label,
            frame_gap=frame_gap,
            plausibility_score=plausibility_score,
            is_large_jump=is_large_jump,
        )
