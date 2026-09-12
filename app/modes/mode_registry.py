"""Registry and JSON persistence for built-in and user-defined modes.

This module is the single lookup boundary used by both CLI and Streamlit.
Adding a built-in mode means adding its factory to ``_BUILTIN_MODES``; custom
modes are loaded from ``AppPaths.mode_config_path``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.config import AppPaths
from app.modes.base_mode import ModeConfig
from app.modes.default_mode import build_default_mode
from app.modes.football_mode import build_football_mode

_BUILTIN_MODES = (
    build_default_mode,
    build_football_mode,
)


def normalize_mode_id(value: str) -> str:
    mode_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", value.strip().lower())
    mode_id = re.sub(r"_+", "_", mode_id).strip("_")
    if not mode_id:
        raise ValueError("Mode id darf nicht leer sein.")
    return mode_id


def custom_modes_path(paths: AppPaths | None = None) -> Path:
    paths = paths or AppPaths()
    paths.ensure()
    return paths.mode_config_path


def builtin_modes() -> dict[str, ModeConfig]:
    modes = [factory() for factory in _BUILTIN_MODES]
    return {mode.mode_id: mode for mode in modes}


def load_custom_modes(paths: AppPaths | None = None) -> dict[str, ModeConfig]:
    path = custom_modes_path(paths)
    if not path.exists():
        return {}

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    modes: dict[str, ModeConfig] = {}
    for item in raw.get("modes", []):
        try:
            mode = ModeConfig.from_json_dict(item)
        except TypeError:
            continue
        mode = ModeConfig.from_json_dict({**mode.to_json_dict(), "is_custom": True})
        modes[mode.mode_id] = mode
    return modes


def list_modes(paths: AppPaths | None = None) -> dict[str, ModeConfig]:
    modes = builtin_modes()
    modes.update(load_custom_modes(paths))
    return modes


def get_mode(mode_id: str, paths: AppPaths | None = None) -> ModeConfig:
    modes = list_modes(paths)
    normalized = normalize_mode_id(mode_id)
    if normalized not in modes:
        available = ", ".join(sorted(modes.keys()))
        raise KeyError(f"Unknown mode '{mode_id}'. Available modes: {available}")
    return modes[normalized]


def save_custom_mode(mode: ModeConfig, paths: AppPaths | None = None, overwrite: bool = False) -> ModeConfig:
    path = custom_modes_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)

    builtins = builtin_modes()
    existing_custom = load_custom_modes(paths)
    mode_id = normalize_mode_id(mode.mode_id)

    if mode_id in builtins:
        raise ValueError("Built-in modes dürfen nicht überschrieben werden.")
    if not overwrite and mode_id in existing_custom:
        raise ValueError(f"Ein Custom Mode mit der ID '{mode_id}' existiert bereits.")

    saved_mode = ModeConfig.from_json_dict({**mode.to_json_dict(), "mode_id": mode_id, "is_custom": True})
    existing_custom[mode_id] = saved_mode

    payload = {
        "modes": [existing_custom[key].to_json_dict() for key in sorted(existing_custom.keys())]
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return saved_mode
