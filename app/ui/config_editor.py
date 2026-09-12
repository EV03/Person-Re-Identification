"""One parameter contract for editing, running and saving a ReID preset.

Only identity/description fields differ between a saved preset and a run. All
runtime parameters are copied together, so saving cannot silently omit a gate.
This module is independent of Streamlit and can be tested without models.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.config import PipelineConfig
from app.modes.base_mode import ModeConfig


_IDENTITY_FIELDS = {"mode_id", "mode_name", "pipeline_type", "is_custom"}
RUNTIME_PARAMETER_FIELDS = tuple(
    name for name in PipelineConfig.__dataclass_fields__ if name not in _IDENTITY_FIELDS
)


def runtime_parameters(config: PipelineConfig) -> dict[str, Any]:
    values = asdict(config)
    return {name: values[name] for name in RUNTIME_PARAMETER_FIELDS}


def changed_parameters(mode: ModeConfig, parameters: dict[str, Any]) -> dict[str, dict[str, Any]]:
    baseline = runtime_parameters(mode.to_pipeline_config())
    return {
        name: {"preset": baseline[name], "current": parameters[name]}
        for name in RUNTIME_PARAMETER_FIELDS
        if baseline[name] != parameters[name]
    }


def build_run_config(mode: ModeConfig, parameters: dict[str, Any]) -> PipelineConfig:
    if set(parameters) != set(RUNTIME_PARAMETER_FIELDS):
        raise ValueError("The editor must supply every runtime parameter exactly once.")
    config = mode.to_pipeline_config(**parameters)
    if changed_parameters(mode, parameters):
        config.mode_name = f"{mode.name} (geändert)"
    return config


def preset_from_config(
    config: PipelineConfig, *, mode_id: str, name: str, description: str
) -> ModeConfig:
    """Save exactly the run's parameters, with a new preset identity."""
    return ModeConfig(
        mode_id=mode_id,
        name=name,
        description=description,
        pipeline_type=config.pipeline_type,
        is_custom=True,
        **runtime_parameters(config),
    )
