from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.storage.models import Detection


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


def save_crop(crop: np.ndarray, snapshot_dir: Path, person_id: str, frame_index: int) -> Path:
    person_dir = snapshot_dir / person_id
    person_dir.mkdir(parents=True, exist_ok=True)
    path = person_dir / f"frame_{frame_index:08d}.jpg"
    cv2.imwrite(str(path), crop)
    return path


def draw_detection(frame: np.ndarray, detection: Detection, person_id: str | None, score: float | None) -> None:
    x1, y1, x2, y2 = detection.bbox_xyxy
    label_parts = [f"track {detection.track_id}"]
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


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_vector(a)
    b = normalize_vector(b)
    return float(np.dot(a, b))
