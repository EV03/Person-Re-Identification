from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from app.config import PipelineConfig, PipelineSettings


@dataclass(frozen=True)
class ModeConfig(PipelineSettings):
    """Configuration preset for a selectable analysis mode.

    Presets configure the same person-ReID pipeline. Built-in presets represent
    B0, A1, A2, A3 and the D1-D6 Details-Tracking family; custom presets support pilot runs.
    """

    mode_id: str
    name: str
    description: str

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
