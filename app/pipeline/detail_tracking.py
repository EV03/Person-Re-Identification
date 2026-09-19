"""Optional identity-decision policy ported from ``Details_Tracking``.

The default main pipeline intentionally keeps its original single-threshold
decision.  Supplying :class:`DetailTrackingPolicy` enables explainable detail
re-ranking, three decision zones, delayed new-person creation and a small
image-space continuity bonus.  Motion is supporting evidence, never a
biometric identifier.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from app.reid.embeddings import checked_embedding, cosine_similarity
from app.reid.repository import IdentityProfile
from app.storage.models import MatchResult
from app.utils.detail_utils import combine_visual_and_detail_score, detail_similarity_breakdown


@dataclass(frozen=True)
class DetailTrackingPolicy:
    detail_weight: float = .15
    detail_min_confidence: float = .55
    strong_match_threshold: float = .82
    weak_match_threshold: float = .68
    new_person_max_score: float = .58
    new_person_min_evidence_events: int = 6
    new_person_min_evidence_span_frames: int = 15
    new_person_evidence_window_frames: int = 30
    new_person_low_match_ratio: float = .80
    new_person_overlap_threshold: float = .65
    motion_identity_bonus: float = .04
    motion_identity_max_frame_gap: int = 15
    motion_identity_max_distance_fraction: float = .15

    def __post_init__(self) -> None:
        for name in ("detail_weight", "detail_min_confidence", "new_person_low_match_ratio",
                     "new_person_overlap_threshold", "motion_identity_bonus",
                     "motion_identity_max_distance_fraction"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1.")
        for name in ("strong_match_threshold", "weak_match_threshold", "new_person_max_score"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not -1 <= value <= 1:
                raise ValueError(f"{name} must be finite and between -1 and 1.")
        if self.strong_match_threshold < self.weak_match_threshold:
            raise ValueError("strong_match_threshold must be at least weak_match_threshold.")
        for name in ("new_person_min_evidence_events", "new_person_min_evidence_span_frames",
                     "new_person_evidence_window_frames", "motion_identity_max_frame_gap"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1.")

    def decision_zone(self, score: float) -> str:
        if score >= self.strong_match_threshold:
            return "strong"
        if score >= self.weak_match_threshold:
            return "weak"
        return "low"

    def as_metadata(self) -> dict[str, float | int | str]:
        return {"name": "details_tracking_v2", **self.__dict__}

    def rank_profiles(
        self,
        embedding: np.ndarray,
        detail_vector: list[float] | tuple[float, ...] | np.ndarray | None,
        profiles: list[IdentityProfile],
        *,
        exclude_person_ids: set[str],
        bbox_xyxy: tuple[int, int, int, int],
        frame_index: int,
        image_width: int,
        image_height: int,
        recent_person_positions: dict[str, tuple[int, tuple[float, float]]],
    ) -> list[MatchResult]:
        query = checked_embedding(embedding)
        center = bbox_center(bbox_xyxy)
        diagonal = max(float(np.hypot(image_width, image_height)), 1.0)
        matches: list[MatchResult] = []
        for profile in profiles:
            if profile.person_id in exclude_person_ids or profile.embedding.shape != query.shape:
                continue
            visual_score = cosine_similarity(query, checked_embedding(profile.embedding))
            breakdown = detail_similarity_breakdown(detail_vector, profile.detail_vector)
            detail_score = float(breakdown["score"]) if detail_vector is not None else None
            final_score = combine_visual_and_detail_score(visual_score, detail_score, self.detail_weight)
            motion_bonus = 0.0
            recent = recent_person_positions.get(profile.person_id)
            if recent is not None:
                previous_frame, previous_center = recent
                frame_gap = int(frame_index) - int(previous_frame)
                distance_fraction = float(np.hypot(center[0] - previous_center[0], center[1] - previous_center[1]) / diagonal)
                if 0 <= frame_gap <= self.motion_identity_max_frame_gap and distance_fraction <= self.motion_identity_max_distance_fraction:
                    motion_bonus = self.motion_identity_bonus
                    final_score = float(np.clip(final_score + motion_bonus, -1.0, 1.0))
            matches.append(MatchResult(
                profile.person_id, final_score, False,
                visual_score=visual_score, detail_score=detail_score,
                detail_weight=self.detail_weight if detail_vector is not None else 0.0,
                detail_breakdown=breakdown, decision_zone=self.decision_zone(final_score),
                motion_bonus=motion_bonus,
            ))
        matches.sort(key=lambda item: item.score, reverse=True)
        top = [
            {"rank": index + 1, "person_id": match.person_id, "score": match.score,
             "visual_score": match.visual_score, "detail_score": match.detail_score,
             "motion_bonus": match.motion_bonus, "decision_zone": match.decision_zone}
            for index, match in enumerate(matches[:5])
        ]
        return [replace(match, top_matches=top) for match in matches]


def bbox_center(bbox_xyxy: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    return ((float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2)


def intersection_over_smaller_box(first: tuple[int, int, int, int],
                                  second: tuple[int, int, int, int]) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    width = max(0, min(first_x2, second_x2) - max(first_x1, second_x1))
    height = max(0, min(first_y2, second_y2) - max(first_y1, second_y1))
    first_area = max(0, first_x2 - first_x1) * max(0, first_y2 - first_y1)
    second_area = max(0, second_x2 - second_x1) * max(0, second_y2 - second_y1)
    denominator = min(first_area, second_area)
    return 0.0 if denominator <= 0 else float(width * height / denominator)


def has_sufficient_new_person_evidence(evidence: list[tuple[int, float]], *, max_score: float,
                                       min_events: int, min_span_frames: int,
                                       low_match_ratio: float) -> bool:
    if not evidence:
        return False
    low_count = sum(1 for _, score in evidence if score <= max_score)
    frame_span = int(evidence[-1][0]) - int(evidence[0][0]) + 1
    return (len(evidence) >= min_events and frame_span >= min_span_frames
            and low_count / len(evidence) >= low_match_ratio)
