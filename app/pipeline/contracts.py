"""Backend-neutral contracts for frame tracking and embedding extraction."""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from app.config import PipelineConfig
from app.storage.models import Detection


class PersonTracker(Protocol):
    """One fresh instance per source; IDs must remain stable within that source.

    Input is an unannotated BGR frame. Return person boxes in pixel XYXY
    coordinates. Missing IDs stay None; do not invent an ID per detection.
    Implementations must not modify the supplied frame.
    """

    def track_frame(self, frame_bgr: np.ndarray) -> list[Detection]: ...


class EmbeddingEncoder(Protocol):
    """Return a finite, normalized, fixed-dimensional embedding from BGR."""

    def encode(self, crop_bgr: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class BackendMetadata(Protocol):
    """Optional adapter-owned metadata; no library internals in orchestration."""

    def describe_backend(self) -> dict[str, Any]: ...


@runtime_checkable
class ReleasableBackend(Protocol):
    """Optional resource lifecycle for backends holding handles or workers."""

    def release(self) -> None: ...


TrackerFactory = Callable[[PipelineConfig], PersonTracker]
