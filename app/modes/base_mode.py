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
    match_threshold: float = 0.82
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
