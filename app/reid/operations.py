"""Pure, replaceable matching and profile-update methods (no SQL or I/O)."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from app.reid.repository import EmbeddingBatch, IdentityProfile
from app.storage.models import MatchResult
from app.reid.embeddings import checked_embedding, cosine_similarity, normalize_vector


class CosineProfileMatcher:
    """Linear cosine search; preserve first candidate on equal scores."""

    def match(self, embedding: np.ndarray, profiles: Iterable[IdentityProfile],
              threshold: float, exclude_person_ids: set[str]) -> MatchResult | None:
        query = normalize_vector(embedding)
        best_person_id: str | None = None
        best_score = float("-inf")
        for profile in profiles:
            if profile.person_id in exclude_person_ids or profile.embedding.shape != query.shape:
                continue
            score = cosine_similarity(query, profile.embedding)
            if score > best_score:
                best_person_id, best_score = profile.person_id, score
        if best_person_id is None or best_score < threshold:
            return None
        return MatchResult(best_person_id, best_score, False)


class WeightedMeanProfileUpdater:
    """Accumulate raw weighted sums; normalize only the derived search profile.

    Initial batches retain every accepted crop's contribution. Legacy profiles
    without a raw sum cannot be reconstructed and must not be silently reused.
    """

    def update(self, previous: IdentityProfile | None, *, person_id: str,
               embedding: np.ndarray, weight: float, snapshot_path: str | None,
               snapshot_quality: float, timestamp: str,
               batch: EmbeddingBatch | None = None) -> IdentityProfile:
        vector = checked_embedding(embedding)
        if batch is None:
            batch = EmbeddingBatch(vector * weight, weight, 1)
        addition = np.asarray(batch.embedding_sum, dtype=np.float64)
        if (addition.shape != vector.shape or not np.isfinite(addition).all()
                or not np.isfinite(batch.weight_sum) or batch.weight_sum <= 0
                or not isinstance(batch.observations, int) or batch.observations < 1):
            raise ValueError("Invalid weighted embedding batch.")
        if not np.allclose(checked_embedding(addition), vector, rtol=1e-5, atol=1e-6):
            raise ValueError("Batch sum and matching embedding must have the same direction.")
        if previous is None:
            total = addition.copy()
            total_weight, count = batch.weight_sum, batch.observations
            created_at, best_path, best_quality = timestamp, snapshot_path, snapshot_quality
        else:
            if previous.embedding.shape != vector.shape:
                raise ValueError("Embedding dimension changed; use the database for this encoder.")
            if previous.embedding_sum is None:
                raise ValueError("Legacy profile has no exact embedding sum; start with a fresh profile database.")
            old_sum = np.asarray(previous.embedding_sum, dtype=np.float64)
            if old_sum.shape != vector.shape or not np.isfinite(old_sum).all():
                raise ValueError("Stored embedding sum is invalid.")
            total = old_sum + addition
            total_weight = previous.embedding_weight_sum + batch.weight_sum
            count = previous.observations + batch.observations
            created_at = previous.created_at
            best_path = previous.best_snapshot_path or snapshot_path
            best_quality = previous.best_snapshot_quality
            if snapshot_path and snapshot_quality >= best_quality:
                best_path, best_quality = snapshot_path, snapshot_quality
        if not np.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("Stored embedding weight sum is invalid.")
        profile_vector = checked_embedding(total / total_weight).astype(np.float32)
        return IdentityProfile(person_id, profile_vector, count, total_weight,
                               created_at, timestamp, best_path, best_quality, total)
