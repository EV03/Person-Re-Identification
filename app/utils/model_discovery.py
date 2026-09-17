from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.config import PROJECT_ROOT


YOLO_EXTENSIONS = {".pt", ".onnx", ".engine"}
OSNET_EXTENSIONS = {".pt", ".pth", ".tar"}
OSNET_MODEL_NAMES = (
    "osnet_ibn_x1_0",
    "osnet_ain_x1_0",
    "osnet_x1_0",
    "osnet_x0_75",
    "osnet_x0_5",
    "osnet_x0_25",
)


@dataclass(frozen=True)
class LocalModel:
    family: str
    name: str
    path: Path
    source: str

    @property
    def display_name(self) -> str:
        return f"{self.name} - {self.source}"


def _environment_directories(name: str) -> list[Path]:
    value = os.getenv(name, "").strip()
    if not value:
        return []
    return [Path(part).expanduser() for part in value.split(os.pathsep) if part.strip()]


def _iter_model_files(directory: Path, extensions: set[str], *, recursive: bool) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        (path.resolve() for path in iterator if path.is_file() and path.suffix.lower() in extensions),
        key=lambda path: path.name.lower(),
    )


def _osnet_name_from_path(path: Path) -> str | None:
    filename = path.name.lower()
    return next((name for name in OSNET_MODEL_NAMES if name in filename), None)


def discover_yolo_models(project_root: Path = PROJECT_ROOT) -> list[LocalModel]:
    project_root = project_root.resolve()
    search_locations: list[tuple[Path, str, bool]] = [
        (project_root / "models" / "yolo", "project models/yolo", True),
        (project_root, "project root", False),
    ]
    search_locations.extend(
        (directory, f"REID_YOLO_MODEL_DIR: {directory}", True)
        for directory in _environment_directories("REID_YOLO_MODEL_DIR")
    )

    models: list[LocalModel] = []
    seen_names: set[str] = set()
    for directory, source, recursive in search_locations:
        for path in _iter_model_files(directory, YOLO_EXTENSIONS, recursive=recursive):
            key = path.name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            models.append(LocalModel(family="yolo", name=path.name, path=path, source=source))
    return models


def discover_osnet_models(project_root: Path = PROJECT_ROOT) -> list[LocalModel]:
    project_root = project_root.resolve()
    search_locations: list[tuple[Path, str, bool]] = [
        (project_root / "models" / "reid", "project models/reid", True),
        (Path.home() / ".cache" / "torch" / "checkpoints", "Torch cache", False),
    ]
    search_locations.extend(
        (directory, f"REID_OSNET_MODEL_DIR: {directory}", True)
        for directory in _environment_directories("REID_OSNET_MODEL_DIR")
    )

    models: list[LocalModel] = []
    seen_names: set[str] = set()
    for directory, source, recursive in search_locations:
        for path in _iter_model_files(directory, OSNET_EXTENSIONS, recursive=recursive):
            model_name = _osnet_name_from_path(path)
            if model_name is None or model_name in seen_names:
                continue
            seen_names.add(model_name)
            models.append(LocalModel(family="osnet", name=model_name, path=path, source=source))
    return models


def find_osnet_model(model_name: str, project_root: Path = PROJECT_ROOT) -> LocalModel | None:
    normalized = str(model_name or "").strip().lower()
    return next((model for model in discover_osnet_models(project_root) if model.name == normalized), None)
