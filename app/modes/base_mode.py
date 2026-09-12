from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import os

from app.config import PipelineConfig


@dataclass(frozen=True)
class ModeConfig:
    """Configuration preset for a selectable analysis mode.

    Presets configure the same person-ReID pipeline. Built-in presets represent
    B0, A1 and A2 from the evaluation plan; custom presets support pilot runs.
    """

    mode_id: str
    name: str
    description: str
    pipeline_type: str = "person_reid"

    yolo_model: str = "yolov8n.pt"
    tracker: str = "bytetrack.yaml"
    encoder_backend: str = "torchreid"
    reid_model_name: str = "osnet_x1_0"
    reid_checkpoint: str = os.getenv("REID_CHECKPOINT", "data/models/osnet_x1_0_msmt17.pth")
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

    is_custom: bool = False

    def __post_init__(self) -> None:
        if self.pipeline_type != "person_reid":
            raise ValueError("Only person_reid presets are supported on this branch.")

    def to_pipeline_config(self, **overrides: Any) -> PipelineConfig:
        values = asdict(self)
        supported_fields = PipelineConfig.__dataclass_fields__.keys()
        values = {key: value for key, value in values.items() if key in supported_fields}
        values["mode_name"] = self.name
        values.update(overrides)
        return PipelineConfig(**values)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "ModeConfig":
        supported_fields = cls.__dataclass_fields__.keys()
        cleaned = {key: value for key, value in data.items() if key in supported_fields}
        return cls(**cleaned)
