"""Embedding arithmetic without image libraries, model frameworks or storage."""

from __future__ import annotations

import numpy as np


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(normalize_vector(a), normalize_vector(b)))


def checked_embedding(vector: np.ndarray) -> np.ndarray:
    """Validate the common encoder boundary, including third-party adapters."""
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("An embedding must be a finite, non-empty one-dimensional vector.")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("An embedding must have a finite, positive norm.")
    return vector / norm
