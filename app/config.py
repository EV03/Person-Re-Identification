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
    db_path: Path = _path_from_env("REID_DB_PATH", "data/db/reid.sqlite3")
    snapshot_dir: Path = _path_from_env("REID_SNAPSHOT_DIR", "data/snapshots")
    output_dir: Path = _path_from_env("REID_OUTPUT_DIR", "data/output")
    mode_dir: Path = _path_from_env("REID_MODE_DIR", "data/modes")
    input_dir: Path = PROJECT_ROOT / "data" / "input"

    @property
    def mode_config_path(self) -> Path:
        return self.mode_dir / "custom_modes.json"

    def ensure(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.mode_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineConfig:
    mode_id: str = "default"
    mode_name: str = "Default ReID MVP"
    pipeline_type: str = "person_reid"

    yolo_model: str = os.getenv("REID_DEFAULT_MODEL", "yolov8n.pt")
    tracker: str = os.getenv("REID_DEFAULT_TRACKER", "bytetrack.yaml")
    encoder_backend: str = os.getenv("REID_DEFAULT_ENCODER", "torchreid")
    match_threshold: float = float(os.getenv("REID_DEFAULT_THRESHOLD", "0.82"))
    detection_confidence: float = 0.35
    image_size: int = 640
    reid_every_n_frames: int = 5
    min_good_frames_before_reid: int = 3
    min_embedding_quality: float = 0.55
    min_update_quality: float = 0.65
    max_frames: int = 500
    min_crop_height: int = 80
    min_crop_width: int = 30
    crop_padding: float = 0.05
    device: str = "auto"
    draw_debug: bool = True
    live_preview_every_n_frames: int = 10
    enable_motion_analysis: bool = True
    draw_motion_vectors: bool = True
    motion_max_jump_fraction: float = 0.20
    motion_smoothing_alpha: float = 0.35
    motion_min_displacement_px: float = 2.0

    enable_ball_tracking: bool = False
    enable_pitch_mapping: bool = False
    enable_team_classification: bool = False
    enable_stats_aggregation: bool = False
    is_custom: bool = False
