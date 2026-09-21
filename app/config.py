"""Central runtime configuration and filesystem locations.

Environment variables are loaded when this module is imported.  ``AppPaths``
describes where persistent artifacts live, while ``PipelineConfig`` contains
the algorithm and runtime settings for one pipeline execution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_from_env(name: str, default: str) -> Path:
    value = os.getenv(name, default)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass(frozen=True)
class AppPaths:
    """Resolved project paths for the database, inputs, outputs and mode files.

    Relative values supplied through environment variables are interpreted
    relative to the repository root.  Call :meth:`ensure` before writing.
    """

    db_path: Path = _path_from_env("REID_DB_PATH", "data/db/reid.sqlite3")
    snapshot_dir: Path = _path_from_env("REID_SNAPSHOT_DIR", "data/snapshots")
    output_dir: Path = _path_from_env("REID_OUTPUT_DIR", "data/output")
    mode_dir: Path = _path_from_env("REID_MODE_DIR", "data/modes")
    input_dir: Path = PROJECT_ROOT / "data" / "input"

    @property
    def mode_config_path(self) -> Path:
        return self.mode_dir / "reid_presets.json"

    def ensure(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.mode_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, kw_only=True)
class PipelineSettings:
    """Single definition of settings shared by saved presets and runtime runs.

    Settings are keyword-only so preset identity can remain positional. Use
    ``dataclasses.replace`` for changes instead of mutating a running config.
    """

    pipeline_type: str = "person_reid"

    yolo_model: str = os.getenv("REID_DEFAULT_MODEL", "yolov8n.pt")
    tracker: str = os.getenv("REID_DEFAULT_TRACKER", "bytetrack.yaml")
    reid_model_name: str = "osnet_x1_0"
    reid_checkpoint: str = os.getenv("REID_CHECKPOINT", "data/models/osnet_x1_0_msmt17.pth")
    match_threshold: float = float(os.getenv("REID_DEFAULT_THRESHOLD", "0.82"))
    detection_confidence: float = 0.35
    image_size: int = 640
    reid_every_n_frames: int = 10
    min_good_frames_before_reid: int = 5
    initial_candidate_every_n_frames: int = 3
    min_embedding_quality: float = 0.55
    min_initial_blur_score: float = 0.40
    min_border_blur_score: float = 0.45
    min_initial_aspect_ratio_score: float = 0.50
    min_update_quality: float = 0.65
    min_update_similarity: float = 0.82
    max_frames: int = 500
    min_crop_height: int = 80
    min_crop_width: int = 30
    crop_padding: float = 0.05
    max_person_overlap_ratio: float = 0.15
    overlap_cooldown_frames: int = 10
    device: str = "auto"
    draw_debug: bool = True
    live_preview_every_n_frames: int = 10

    is_custom: bool = False

    def __post_init__(self) -> None:
        if self.pipeline_type != "person_reid":
            raise ValueError("Only the person_reid pipeline is supported on this branch.")
        import math
        for name in ("match_threshold", "min_update_similarity"):
            value = getattr(self, name)
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError(f"{name} must be finite and between -1 and 1.")
        if not math.isfinite(self.max_person_overlap_ratio) or not 0 <= self.max_person_overlap_ratio <= 1:
            raise ValueError("max_person_overlap_ratio must be finite and between 0 and 1.")
        for name in (
            "min_initial_blur_score",
            "min_border_blur_score",
            "min_initial_aspect_ratio_score",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between 0 and 1.")
        if self.overlap_cooldown_frames < 0:
            raise ValueError("overlap_cooldown_frames must be greater than or equal to zero.")
        for name in ("reid_every_n_frames", "min_good_frames_before_reid", "initial_candidate_every_n_frames"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be greater than or equal to one.")


@dataclass(frozen=True)
class PipelineConfig(PipelineSettings):
    """Effective settings plus the identity of one configured run."""

    mode_id: str = "default"
    mode_name: str = "Default ReID MVP"
