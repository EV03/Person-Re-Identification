from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.config import PipelineConfig


@dataclass(frozen=True)
class ModeConfig:
    """Configuration preset for a selectable analysis mode.

    A mode describes the intent of a run and provides defaults for the existing
    pipeline. The actual processing code can stay small while new modes can add
    their own flags and future modules step by step.
    """

    mode_id: str
    name: str
    description: str
    pipeline_type: str = "person_reid"

    yolo_model: str = "yolov8n.pt"
    tracker: str = "bytetrack.yaml"
    encoder_backend: str = "torchreid"
    vector_store_backend: str = "sqlite"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "person_reid_embeddings"
    qdrant_mode: str = "local"
    qdrant_local_path: str = "data/qdrant_local"
    qdrant_prefer_grpc: bool = False
    match_threshold: float = 0.74
    strong_match_threshold: float = 0.82
    weak_match_threshold: float = 0.68
    new_person_max_score: float = 0.58
    new_person_min_evidence_events: int = 6
    new_person_evidence_window_frames: int = 30
    new_person_low_match_ratio: float = 0.80
    pending_max_age_frames: int = 45
    motion_identity_bonus: float = 0.04
    motion_identity_max_frame_gap: int = 15
    motion_identity_max_distance_fraction: float = 0.15
    merge_candidate_threshold: float = 0.86
    merge_candidate_min_events: int = 8
    merge_candidate_same_track_required: bool = True
    detection_confidence: float = 0.40
    image_size: int = 960
    reid_every_n_frames: int = 5
    min_good_frames_before_reid: int = 3
    min_embedding_quality: float = 0.60
    min_update_quality: float = 0.75
    max_frames: int = 500
    min_crop_height: int = 120
    min_crop_width: int = 45
    crop_padding: float = 0.08
    calibration_detection_confidence: float = 0.45
    calibration_image_size: int = 1280
    calibration_reid_every_n_frames: int = 3
    calibration_min_good_frames_before_reid: int = 5
    calibration_min_embedding_quality: float = 0.70
    calibration_min_update_quality: float = 0.80
    calibration_min_crop_height: int = 160
    calibration_min_crop_width: int = 60
    calibration_crop_padding: float = 0.10
    calibration_enable_detail_analysis: bool = False
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
    detail_weight: float = 0.05
    detail_min_confidence: float = 0.70
    draw_detail_labels: bool = True

    enable_ball_tracking: bool = False
    enable_pitch_mapping: bool = False
    enable_team_classification: bool = False
    enable_stats_aggregation: bool = False

    is_custom: bool = False

    def to_pipeline_config(self, **overrides: Any) -> PipelineConfig:
        values = asdict(self)
        supported_fields = PipelineConfig.__dataclass_fields__.keys()
        values = {key: value for key, value in values.items() if key in supported_fields}
        values.update(overrides)
        return PipelineConfig(**values)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "ModeConfig":
        supported_fields = cls.__dataclass_fields__.keys()
        cleaned = {key: value for key, value in data.items() if key in supported_fields}
        return cls(**cleaned)
