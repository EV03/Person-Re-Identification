from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_RUNTIME_DIR = PROJECT_ROOT / ".runtime"
PROJECT_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_RUNTIME_DIR))


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
    reid_model_name: str = os.getenv("REID_DEFAULT_REID_MODEL", "osnet_x1_0")
    reid_model_path: str = os.getenv("REID_DEFAULT_REID_MODEL_PATH", "")
    vector_store_backend: str = os.getenv("REID_VECTOR_STORE_BACKEND", "sqlite")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "person_reid_embeddings")
    qdrant_mode: str = os.getenv("QDRANT_MODE", "local")
    qdrant_local_path: str = os.getenv("QDRANT_LOCAL_PATH", "data/qdrant_local")
    qdrant_prefer_grpc: bool = os.getenv("QDRANT_PREFER_GRPC", "false").lower() in {"1", "true", "yes", "on"}
    # ReID matching uses three zones:
    # strong >= strong_match_threshold -> normal match and safe profile update
    # weak   >= weak_match_threshold   -> tentative/pending match, no direct new ID
    # low    <  weak_match_threshold   -> new ID only after repeated evidence
    match_threshold: float = float(os.getenv("REID_DEFAULT_THRESHOLD", "0.74"))
    strong_match_threshold: float = float(os.getenv("REID_STRONG_MATCH_THRESHOLD", "0.82"))
    weak_match_threshold: float = float(os.getenv("REID_WEAK_MATCH_THRESHOLD", "0.68"))
    new_person_max_score: float = float(os.getenv("REID_NEW_PERSON_MAX_SCORE", "0.58"))
    new_person_min_evidence_events: int = int(os.getenv("REID_NEW_PERSON_MIN_EVIDENCE_EVENTS", "6"))
    new_person_min_evidence_span_frames: int = int(os.getenv("REID_NEW_PERSON_MIN_EVIDENCE_SPAN_FRAMES", "15"))
    new_person_evidence_window_frames: int = int(os.getenv("REID_NEW_PERSON_EVIDENCE_WINDOW_FRAMES", "30"))
    new_person_low_match_ratio: float = float(os.getenv("REID_NEW_PERSON_LOW_MATCH_RATIO", "0.80"))
    new_person_overlap_threshold: float = float(os.getenv("REID_NEW_PERSON_OVERLAP_THRESHOLD", "0.65"))
    pending_max_age_frames: int = int(os.getenv("REID_PENDING_MAX_AGE_FRAMES", "45"))
    motion_identity_bonus: float = float(os.getenv("REID_MOTION_IDENTITY_BONUS", "0.04"))
    motion_identity_max_frame_gap: int = int(os.getenv("REID_MOTION_IDENTITY_MAX_FRAME_GAP", "15"))
    motion_identity_max_distance_fraction: float = float(os.getenv("REID_MOTION_IDENTITY_MAX_DISTANCE_FRACTION", "0.15"))
    merge_candidate_threshold: float = float(os.getenv("REID_MERGE_CANDIDATE_THRESHOLD", "0.86"))
    merge_candidate_min_events: int = int(os.getenv("REID_MERGE_CANDIDATE_MIN_EVENTS", "8"))
    merge_candidate_same_track_required: bool = os.getenv("REID_MERGE_SAME_TRACK_REQUIRED", "true").lower() in {"1", "true", "yes", "on"}
    detection_confidence: float = 0.40
    image_size: int = 1280
    reid_every_n_frames: int = 5
    min_good_frames_before_reid: int = 3
    min_embedding_quality: float = 0.55
    min_update_quality: float = 0.65
    max_frames: int = 500
    min_crop_height: int = 80
    min_crop_width: int = 30
    crop_padding: float = 0.05
    calibration_mode: str = "off"
    calibration_target_person_id: str = ""
    calibration_label: str = ""
    device: str = "auto"
    draw_debug: bool = True
    live_preview_every_n_frames: int = 10
    enable_motion_analysis: bool = True
    draw_motion_vectors: bool = True
    disable_internal_motion_when_botsort: bool = False
    motion_max_jump_fraction: float = 0.20
    motion_smoothing_alpha: float = 0.35
    motion_min_displacement_px: float = 2.0
    enable_detail_analysis: bool = True
    detail_weight: float = 0.15
    detail_min_confidence: float = 0.55
    draw_detail_labels: bool = True

    enable_ball_tracking: bool = False
    enable_pitch_mapping: bool = False
    enable_team_classification: bool = False
    enable_stats_aggregation: bool = False
    is_custom: bool = False
