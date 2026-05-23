from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    track_id: int
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    class_id: int = 0


@dataclass(frozen=True)
class MatchResult:
    person_id: str
    score: float
    is_new: bool


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


@dataclass(frozen=True)
class FootballPlayerFrameEvent:
    run_id: str
    frame_index: int
    timestamp_sec: float
    track_id: int
    player_id: str | None
    team_id: str | None
    bbox_xyxy: tuple[int, int, int, int]
    confidence: float
    pitch_x: float | None = None
    pitch_y: float | None = None
    speed_mps: float | None = None
    distance_delta_m: float | None = None


@dataclass(frozen=True)
class FootballBallFrameEvent:
    run_id: str
    frame_index: int
    timestamp_sec: float
    bbox_xyxy: tuple[int, int, int, int] | None
    confidence: float | None = None
    pitch_x: float | None = None
    pitch_y: float | None = None
    speed_mps: float | None = None
    nearest_player_id: str | None = None
    nearest_team_id: str | None = None


Embedding = np.ndarray
Payload = dict[str, Any]
