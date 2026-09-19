"""Compose interchangeable matching/update policies with a profile repository."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.reid.repository import EmbeddingBatch, ProfileMatcher, ProfileObservation, ProfileRepository, ProfileSearchDecision, ProfileUpdateDecision, ProfileUpdater
from app.reid.operations import CosineProfileMatcher, WeightedMeanProfileUpdater
from app.reid.embeddings import checked_embedding, cosine_similarity
from app.storage.models import MatchResult, Payload, PersonRecord
from app.utils.id_utils import utc_now_iso


def _bounded_payload_value(value: Any, lower: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        raise ValueError("Embedding quality/weight must be finite.")
    return float(np.clip(numeric, lower, 1.0))


class ProfileService:
    """Identity operations for a sequential research run, independent of SQLite."""

    def __init__(self, repository: ProfileRepository, *, matcher: ProfileMatcher | None = None,
                 updater: ProfileUpdater | None = None, min_update_similarity: float = .82) -> None:
        if not np.isfinite(min_update_similarity) or not -1 <= min_update_similarity <= 1:
            raise ValueError("Update similarity threshold must be finite and between -1 and 1.")
        self.repository = repository
        self.matcher = matcher if matcher is not None else CosineProfileMatcher()
        self.updater = updater if updater is not None else WeightedMeanProfileUpdater()
        self.min_update_similarity = min_update_similarity

    def describe_backend(self) -> dict[str, Any]:
        def name(component: object) -> str:
            return f"{type(component).__module__}.{type(component).__qualname__}"
        return {"repository": name(self.repository), "matcher": name(self.matcher), "updater": name(self.updater),
                "min_update_similarity": self.min_update_similarity}

    def search(self, embedding: np.ndarray, threshold: float,
               exclude_person_ids: set[str] | None = None) -> MatchResult | None:
        return self.search_with_diagnostics(embedding, threshold, exclude_person_ids).match

    def search_with_diagnostics(self, embedding: np.ndarray, threshold: float,
                                exclude_person_ids: set[str] | None = None) -> ProfileSearchDecision:
        query = checked_embedding(embedding)
        excluded = exclude_person_ids or set()
        profiles = list(self.repository.iter_profiles())
        evaluate = getattr(self.matcher, "evaluate", None)
        if callable(evaluate):
            return evaluate(query, profiles, threshold, excluded)

        # Custom matchers written against the original small interface remain
        # supported. Their internal rejected score is unknowable, but the run
        # still records why no ordinary candidate was available.
        match = self.matcher.match(query, profiles, threshold, excluded)
        eligible = sum(
            profile.person_id not in excluded and profile.embedding.shape == query.shape
            for profile in profiles
        )
        if match is not None:
            return ProfileSearchDecision(
                match, match.person_id, float(match.score), "matched_existing_profile",
                eligible, tuple(sorted(excluded)),
            )
        if not profiles:
            reason = "empty_database"
        elif eligible == 0:
            reason = "no_eligible_profile"
        else:
            reason = "matcher_rejected"
        return ProfileSearchDecision(None, None, None, reason, eligible, tuple(sorted(excluded)))

    def create_person_id(self) -> str:
        return self.repository.create_person_id()

    def list_persons(self) -> list[PersonRecord]:
        return self.repository.list_persons()

    def add_or_update_person(self, person_id: str, embedding: np.ndarray, source: str,
                             frame_index: int, track_id: int, bbox_xyxy: tuple[int, int, int, int],
                             score: float | None, snapshot_path: str | None,
                             payload: Payload | None = None,
                             batch: EmbeddingBatch | None = None) -> ProfileUpdateDecision:
        payload = dict(payload or {})
        timestamp = utc_now_iso()
        weight = _bounded_payload_value(payload.get("embedding_weight", payload.get("quality_score", 1.0)), .05, 1.0)
        quality = _bounded_payload_value(payload.get("snapshot_quality", payload.get("quality_score", 0.0)), 0.0, 0.0)
        payload["embedding_weight"] = batch.weight_sum if batch is not None else weight
        payload["embedding_observations"] = batch.observations if batch is not None else 1
        embedding = checked_embedding(embedding)
        previous = self.repository.get_profile(person_id)
        similarity = None
        if previous is not None:
            if previous.embedding.shape != embedding.shape:
                raise ValueError("Embedding dimension changed; use a separate encoder database.")
            similarity = float(np.clip(cosine_similarity(checked_embedding(previous.embedding), embedding), -1, 1))
        accepted = similarity is None or similarity >= self.min_update_similarity
        decision = ProfileUpdateDecision(accepted, similarity,
                                         "below_update_similarity" if not accepted else "created_profile" if previous is None else "updated_profile")
        payload.update(profile_update_accepted=accepted, update_similarity=similarity,
                       min_update_similarity=self.min_update_similarity, profile_update_reason=decision.reason)
        if not accepted:
            payload["event_type"] = "profile_update_rejected"
            observation = ProfileObservation(person_id, source, frame_index, track_id, bbox_xyxy,
                                             score, snapshot_path, timestamp, payload)
            self.repository.save_observation(previous, observation)
            return decision
        profile = self.updater.update(previous, person_id=person_id,
                                      embedding=embedding, weight=weight, snapshot_path=snapshot_path,
                                      snapshot_quality=quality, timestamp=timestamp, batch=batch)
        observation = ProfileObservation(person_id, source, frame_index, track_id, bbox_xyxy,
                                         score, snapshot_path, timestamp, payload)
        self.repository.save_observation(profile, observation)
        return decision
