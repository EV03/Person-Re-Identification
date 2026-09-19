"""Storage-neutral profile state and interchangeable ReID contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from app.storage.models import MatchResult, Payload, PersonRecord


@dataclass(frozen=True)
class IdentityProfile:
    person_id: str
    embedding: np.ndarray
    observations: int
    embedding_weight_sum: float
    created_at: str
    last_seen: str
    best_snapshot_path: str | None = None
    best_snapshot_quality: float = 0.0
    embedding_sum: np.ndarray | None = None
    detail_vector: np.ndarray | None = None
    detail_weight_sum: float = 0.0


@dataclass(frozen=True)
class EmbeddingBatch:
    """Unnormalized weighted sum of accepted normalized crop embeddings."""

    embedding_sum: np.ndarray
    weight_sum: float
    observations: int


@dataclass(frozen=True)
class ProfileUpdateDecision:
    accepted: bool
    similarity: float | None
    reason: str


@dataclass(frozen=True)
class ProfileObservation:
    person_id: str
    source: str
    frame_index: int
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    score: float | None
    snapshot_path: str | None
    created_at: str
    payload: Payload


class ProfileRepository(Protocol):
    """Persist state, not matching/update algorithms.

    save_observation must atomically save the profile and its event. The current
    service assumes sequential writes (one evaluation worker per repository).
    """

    def get_profile(self, person_id: str) -> IdentityProfile | None: ...
    def iter_profiles(self) -> Iterable[IdentityProfile]: ...
    def save_observation(self, profile: IdentityProfile, observation: ProfileObservation) -> None: ...
    def create_person_id(self) -> str: ...
    def list_persons(self) -> list[PersonRecord]: ...


class ProfileMatcher(Protocol):
    def match(self, embedding: np.ndarray, profiles: Iterable[IdentityProfile],
              threshold: float, exclude_person_ids: set[str]) -> MatchResult | None: ...


class ProfileUpdater(Protocol):
    def update(self, previous: IdentityProfile | None, *, person_id: str,
               embedding: np.ndarray, weight: float, snapshot_path: str | None,
               snapshot_quality: float, timestamp: str,
               batch: EmbeddingBatch | None = None,
               detail_vector: np.ndarray | None = None,
               detail_weight: float = 0.0) -> IdentityProfile: ...


class ProfileManager(Protocol):
    """Application-facing matching and observation operations."""

    def search(self, embedding: np.ndarray, threshold: float,
               exclude_person_ids: set[str] | None = None) -> MatchResult | None: ...
    def create_person_id(self) -> str: ...
    def add_or_update_person(self, person_id: str, embedding: np.ndarray, source: str,
                             frame_index: int, track_id: int, bbox_xyxy: tuple[int, int, int, int],
                             score: float | None, snapshot_path: str | None,
                             payload: Payload | None = None,
                             batch: EmbeddingBatch | None = None) -> ProfileUpdateDecision: ...
    def list_persons(self) -> list[PersonRecord]: ...


class RunRepository(Protocol):
    """Run bookkeeping is separate from identity decisions."""

    def add_analysis_run(self, run_id: str, mode_id: str, mode_name: str, pipeline_type: str,
                         source: str, fps: float | None, frame_count: int | None,
                         width: int | None, height: int | None, metadata: Payload | None = None) -> None: ...
    def finish_analysis_run(self, run_id: str, *, status: str,
                            processed_frames: int, error: str | None) -> None: ...


class EvaluationRepository(ProfileRepository, RunRepository, Protocol):
    """Default combined persistence adapter; individual contracts stay separate."""
