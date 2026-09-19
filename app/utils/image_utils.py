from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.storage.models import Detection
from app.reid.embeddings import cosine_similarity, normalize_vector  # Compatibility exports.


def crop_xyxy(frame: np.ndarray, bbox_xyxy: tuple[int, int, int, int], padding: float = 0.0) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy

    box_w = max(0, x2 - x1)
    box_h = max(0, y2 - y1)
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def is_valid_crop(crop: np.ndarray | None, min_width: int, min_height: int) -> bool:
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    return w >= min_width and h >= min_height


def blur_score(crop: np.ndarray | None) -> float:
    """Return a normalized sharpness score in the range 0..1.

    The raw Laplacian variance is intentionally compressed into a simple MVP
    score. It is not a classifier, but good enough to reject obviously blurry
    crops before they pollute the ReID database.
    """
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.clip(variance / 150.0, 0.0, 1.0))


def brightness_score(crop: np.ndarray | None) -> float:
    """Return a normalized brightness score in the range 0..1.

    A score close to 1 means the crop is neither very dark nor heavily
    overexposed. The function deliberately does not alter the image; it only
    decides whether the crop is safe enough for ReID storage.
    """
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    mean = float(gray.mean())
    return float(np.clip(1.0 - abs(mean - 128.0) / 128.0, 0.0, 1.0))


def aspect_ratio_score(crop: np.ndarray | None) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    if w <= 0 or h <= 0:
        return 0.0
    ratio = h / max(w, 1)
    if 1.5 <= ratio <= 4.0:
        return 1.0
    if 1.1 <= ratio < 1.5:
        return float((ratio - 1.1) / 0.4)
    if 4.0 < ratio <= 5.0:
        return float(1.0 - (ratio - 4.0) / 1.0)
    return 0.0


def crop_size_score(crop: np.ndarray | None, min_width: int, min_height: int) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    width_score = min(1.0, w / max(float(min_width * 2), 1.0))
    height_score = min(1.0, h / max(float(min_height * 2), 1.0))
    return float(min(width_score, height_score))


def person_bbox_overlap_ratio(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    """Return intersection area relative to the smaller person box.

    Using the smaller box instead of union area catches cases where one person
    substantially occludes another even if their boxes differ greatly in size.
    """
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    intersection_width = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection_area = intersection_width * intersection_height
    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    smaller_area = min(first_area, second_area)
    if smaller_area <= 0:
        return 0.0
    return float(np.clip(intersection_area / smaller_area, 0.0, 1.0))


def edge_cutoff_score(bbox_xyxy: tuple[int, int, int, int], frame_shape: tuple[int, ...], margin: int = 3) -> float:
    """Estimate whether the person box touches the frame border.

    Border contact often means the person is only partially visible. The score
    is only a soft penalty because side-entry/exit frames can still be useful
    for drawing and tracking, but should be less trusted for embedding updates.
    """
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    touches_border = x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin
    return 0.65 if touches_border else 1.0


def crop_quality_score(
    crop: np.ndarray | None,
    bbox_xyxy: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    detection_confidence: float,
    min_width: int,
    min_height: int,
) -> tuple[float, dict[str, float]]:
    """Compute a lightweight ReID crop quality score.

    The score is used as a gate before encoding/storing embeddings. It combines
    sharpness, brightness, crop size, body-like aspect ratio, edge cut-off and
    YOLO confidence. All components are intentionally explainable and cheap.
    """
    if crop is None or crop.size == 0:
        details = {
            "blur": 0.0,
            "brightness": 0.0,
            "size": 0.0,
            "aspect_ratio": 0.0,
            "edge_cutoff": 0.0,
            "detection_confidence": float(detection_confidence),
        }
        return 0.0, details

    details = {
        "blur": blur_score(crop),
        "brightness": brightness_score(crop),
        "size": crop_size_score(crop, min_width=min_width, min_height=min_height),
        "aspect_ratio": aspect_ratio_score(crop),
        "edge_cutoff": edge_cutoff_score(bbox_xyxy, frame_shape),
        "detection_confidence": float(np.clip(detection_confidence, 0.0, 1.0)),
    }

    score = (
        details["blur"] * 0.25
        + details["brightness"] * 0.15
        + details["size"] * 0.25
        + details["aspect_ratio"] * 0.15
        + details["edge_cutoff"] * 0.10
        + details["detection_confidence"] * 0.10
    )
    return float(np.clip(score, 0.0, 1.0)), details


def save_crop(crop: np.ndarray, snapshot_dir: Path, person_id: str, frame_index: int, *, run_id: str) -> Path:
    import re

    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", person_id):
        raise ValueError("Snapshot run/person IDs must be plain identifiers, not paths.")
    snapshot_dir = snapshot_dir / run_id
    person_dir = snapshot_dir / person_id
    person_dir.mkdir(parents=True, exist_ok=True)
    path = person_dir / f"frame_{frame_index:08d}.jpg"
    if not cv2.imwrite(str(path), crop):
        raise OSError(f"Could not write person snapshot: {path}")
    return path


def draw_detection(
    frame: np.ndarray,
    detection: Detection,
    person_id: str | None,
    score: float | None,
) -> None:
    x1, y1, x2, y2 = detection.bbox_xyxy
    label_parts = [f"track {detection.track_id}" if detection.track_id is not None else "untracked"]
    if person_id:
        label_parts.append(person_id)
    if score is not None:
        label_parts.append(f"match {score:.2f}")
    label_parts.append(f"det {detection.confidence:.2f}")
    label = " | ".join(label_parts)

    color = (0, 220, 120)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_x1 = max(0, x1)
    label_y1 = max(text_h + baseline + 8, y1)
    label_x2 = min(frame.shape[1] - 1, label_x1 + text_w + 10)
    label_y2 = min(frame.shape[0] - 1, label_y1 + text_h + baseline + 8)

    cv2.rectangle(frame, (label_x1, label_y1 - text_h - baseline - 8), (label_x2, label_y1), color, -1)
    cv2.putText(
        frame,
        label,
        (label_x1 + 5, label_y1 - 6),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
