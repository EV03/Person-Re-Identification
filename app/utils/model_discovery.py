"""Discover locally installed detector and ReID weights for the Streamlit UI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.config import PROJECT_ROOT


YOLO_EXTENSIONS = {".pt", ".onnx", ".engine"}
REID_EXTENSIONS = {".pt", ".pth", ".tar", ".ckpt"}
OSNET_MODEL_NAMES = (
    "osnet_ibn_x1_0", "osnet_ain_x1_0", "osnet_x1_0",
    "osnet_x0_75", "osnet_x0_5", "osnet_x0_25",
)


@dataclass(frozen=True)
class LocalModel:
    family: str
    name: str
    path: Path
    source: str

    def config_path(self, project_root: Path = PROJECT_ROOT) -> str:
        resolved = self.path.resolve()
        try:
            return resolved.relative_to(project_root.resolve()).as_posix()
        except ValueError:
            return str(resolved)

    @property
    def display_name(self) -> str:
        return f"{self.name} · {self.source}"


def _environment_directories(name: str) -> list[Path]:
    value = os.getenv(name, "").strip()
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part.strip()] if value else []


def _iter_model_files(directory: Path, extensions: set[str], *, recursive: bool) -> list[Path]:
    if not directory.is_dir():
        return []
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        (path.resolve() for path in iterator if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: (path.name.lower(), str(path).lower()),
    )


def _discover(locations: list[tuple[Path, str, bool]], extensions: set[str], family: str) -> list[LocalModel]:
    models: list[LocalModel] = []
    seen: set[Path] = set()
    for directory, source, recursive in locations:
        for path in _iter_model_files(directory, extensions, recursive=recursive):
            if path in seen:
                continue
            seen.add(path)
            models.append(LocalModel(family, path.name, path, source))
    return models


def discover_yolo_models(project_root: Path = PROJECT_ROOT) -> list[LocalModel]:
    project_root = project_root.resolve()
    locations = [
        (project_root / "models" / "yolo", "models/yolo", True),
        (project_root / "data" / "models", "data/models", True),
        (project_root, "project root", False),
    ]
    locations.extend((directory, f"REID_YOLO_MODEL_DIR: {directory}", True)
                     for directory in _environment_directories("REID_YOLO_MODEL_DIR"))
    return _discover(locations, YOLO_EXTENSIONS, "yolo")


def osnet_name_from_path(path: Path) -> str | None:
    filename = path.name.lower()
    return next((name for name in OSNET_MODEL_NAMES if name in filename), None)


def discover_reid_models(project_root: Path = PROJECT_ROOT) -> list[LocalModel]:
    project_root = project_root.resolve()
    locations = [
        (project_root / "models" / "reid", "models/reid", True),
        (project_root / "data" / "models", "data/models", True),
        (Path.home() / ".cache" / "torch" / "checkpoints", "Torch cache", False),
    ]
    locations.extend((directory, f"REID_OSNET_MODEL_DIR: {directory}", True)
                     for directory in _environment_directories("REID_OSNET_MODEL_DIR"))
    return _discover(locations, REID_EXTENSIONS, "reid")


def reid_architectures(models: list[LocalModel]) -> list[str]:
    return list(dict.fromkeys(name for model in models if (name := osnet_name_from_path(model.path))))


def selectable_paths(current: str, models: list[LocalModel], project_root: Path = PROJECT_ROOT) -> list[str]:
    """Keep a saved/custom value selectable and append every discovered path."""
    return list(dict.fromkeys([str(current), *(model.config_path(project_root) for model in models)]))


def display_labels(models: list[LocalModel], project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    return {model.config_path(project_root): model.display_name for model in models}
