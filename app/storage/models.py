from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    track_id: int | None
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    class_id: int = 0


@dataclass(frozen=True)
class MatchResult:
    person_id: str
    score: float
    is_new: bool
    visual_score: float | None = None
    detail_score: float | None = None
    detail_weight: float = 0.0
    detail_breakdown: dict[str, Any] = field(default_factory=dict)
    decision_zone: str = "unknown"
    motion_bonus: float = 0.0
    top_matches: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PersonRecord:
    person_id: str
    observations: int
    created_at: str
    last_seen: str
    best_snapshot_path: str | None = None
    last_score: float | None = None


@dataclass
class PipelineResult:
    output_video_path: Path | None
    processed_frames: int
    created_persons: int
    matched_events: int
    persons: list[PersonRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    run_id: str | None = None
    mode_id: str = "default"
    mode_name: str = "Default ReID MVP"
    pipeline_type: str = "person_reid"
    predictions_path: Path | None = None
    tracking_predictions_path: Path | None = None
    manifest_path: Path | None = None
    decision_policy: str = "main_single_threshold"
    strong_match_events: int = 0
    pending_weak_match_events: int = 0
    pending_new_person_events: int = 0


Embedding = np.ndarray
Payload = dict[str, Any]
