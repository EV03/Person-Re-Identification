"""Pure, replaceable matching and profile-update methods (no SQL or I/O)."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from app.reid.repository import EmbeddingBatch, IdentityProfile, ProfileSearchDecision
from app.storage.models import MatchResult
from app.reid.embeddings import checked_embedding, cosine_similarity, normalize_vector


class CosineProfileMatcher:
    """Linear cosine search; preserve first candidate on equal scores."""

    def evaluate(self, embedding: np.ndarray, profiles: Iterable[IdentityProfile],
                 threshold: float, exclude_person_ids: set[str]) -> ProfileSearchDecision:
        query = normalize_vector(embedding)
        profile_list = list(profiles)
        best_person_id: str | None = None
        best_score = float("-inf")
        eligible_profile_count = 0
        for profile in profile_list:
            if profile.person_id in exclude_person_ids or profile.embedding.shape != query.shape:
                continue
            eligible_profile_count += 1
            score = cosine_similarity(query, profile.embedding)
            if score > best_score:
                best_person_id, best_score = profile.person_id, score

        excluded = tuple(sorted(exclude_person_ids))
        if not profile_list:
            return ProfileSearchDecision(None, None, None, "empty_database", 0, excluded)
        if eligible_profile_count == 0:
            return ProfileSearchDecision(None, None, None, "no_eligible_profile", 0, excluded)
        if best_score < threshold:
            return ProfileSearchDecision(
                None, best_person_id, float(best_score), "below_match_threshold",
                eligible_profile_count, excluded,
            )
        match = MatchResult(best_person_id, float(best_score), False)
        return ProfileSearchDecision(
            match, best_person_id, float(best_score), "matched_existing_profile",
            eligible_profile_count, excluded,
        )

    def match(self, embedding: np.ndarray, profiles: Iterable[IdentityProfile],
              threshold: float, exclude_person_ids: set[str]) -> MatchResult | None:
        return self.evaluate(embedding, profiles, threshold, exclude_person_ids).match


class WeightedMeanProfileUpdater:
    """Accumulate raw weighted sums; normalize only the derived search profile.

    Initial batches retain every accepted crop's contribution. Legacy profiles
    without a raw sum cannot be reconstructed and must not be silently reused.
    """

    def update(self, previous: IdentityProfile | None, *, person_id: str,
               embedding: np.ndarray, weight: float, snapshot_path: str | None,
               snapshot_quality: float, timestamp: str,
               batch: EmbeddingBatch | None = None,
               detail_vector: np.ndarray | None = None,
               detail_weight: float = 0.0) -> IdentityProfile:
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
        incoming_detail = None
        incoming_detail_weight = 0.0
        if detail_vector is not None:
            incoming_detail = np.asarray(detail_vector, dtype=np.float64).reshape(-1)
            incoming_detail_weight = float(detail_weight)
            if (incoming_detail.size == 0 or not np.isfinite(incoming_detail).all()
                    or not np.isfinite(incoming_detail_weight) or incoming_detail_weight <= 0):
                raise ValueError("Invalid detail vector or weight.")
        if previous is None:
            total = addition.copy()
            total_weight, count = batch.weight_sum, batch.observations
            created_at, best_path, best_quality = timestamp, snapshot_path, snapshot_quality
            merged_detail = incoming_detail
            merged_detail_weight = incoming_detail_weight
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
            merged_detail = previous.detail_vector
            merged_detail_weight = float(previous.detail_weight_sum)
            if incoming_detail is not None:
                if merged_detail is not None and merged_detail.shape == incoming_detail.shape and merged_detail_weight > 0:
                    total_detail_weight = merged_detail_weight + incoming_detail_weight
                    merged_detail = (merged_detail * merged_detail_weight + incoming_detail * incoming_detail_weight) / total_detail_weight
                    merged_detail_weight = total_detail_weight
                else:
                    merged_detail, merged_detail_weight = incoming_detail, incoming_detail_weight
        if not np.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("Stored embedding weight sum is invalid.")
        profile_vector = checked_embedding(total / total_weight).astype(np.float32)
        return IdentityProfile(person_id, profile_vector, count, total_weight,
                               created_at, timestamp, best_path, best_quality, total,
                               merged_detail, merged_detail_weight)
