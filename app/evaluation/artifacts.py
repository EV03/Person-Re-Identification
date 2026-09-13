"""Streaming frame predictions and an atomic, failure-aware run manifest.

JSONL is the authoritative export: one record per decoded/processed frame,
including empty frames and detections without track/person IDs. The MOT file is
a convenience view of only the predictions with a genuine tracker ID.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.config import AppPaths, PipelineConfig, PROJECT_ROOT
from app.utils.id_utils import utc_now_iso
from app.pipeline.model_weights import DEFAULT_CHECKPOINT_SHA256, DEFAULT_CHECKPOINT_PROVENANCE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_file(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def file_reference(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path) if path.is_file() else None}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Replace a generated artifact atomically; never touch user source files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def code_reference() -> dict[str, Any]:
    git_args = ["git", "-c", f"safe.directory={PROJECT_ROOT.as_posix()}"]
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(git_args + list(args), cwd=PROJECT_ROOT,
                                    capture_output=True, text=True, timeout=10)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None
    # A Git commit alone does not identify an uncommitted prototype. Include
    # the actual source tree, including new Python files, but no data or .env.
    digest = hashlib.sha256()
    sources = [path for folder in ("app", "scripts", "tests")
               for path in (PROJECT_ROOT / folder).rglob("*.py")]
    sources.extend(PROJECT_ROOT.glob("requirements*.txt"))
    for path in sorted(sources):
        digest.update(path.relative_to(PROJECT_ROOT).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None,
            "source_tree_sha256": digest.hexdigest()}


def tracker_reference(tracker: str) -> dict[str, Any]:
    """Resolve a configured YAML for the Ultralytics adapter."""
    tracker_path = resolve_project_file(tracker)
    if not tracker_path.is_file():
        spec = importlib.util.find_spec("ultralytics")
        if spec and spec.origin:
            tracker_path = Path(spec.origin).parent / "cfg" / "trackers" / tracker
    return file_reference(tracker_path)


def model_references(config: PipelineConfig) -> dict[str, Any]:
    checkpoint = (file_reference(resolve_project_file(config.reid_checkpoint))
                  if config.encoder_backend == "torchreid" and config.reid_checkpoint else None)
    if checkpoint:
        checkpoint["provenance"] = (DEFAULT_CHECKPOINT_PROVENANCE
                                    if checkpoint["sha256"] == DEFAULT_CHECKPOINT_SHA256
                                    else {"note": "User-supplied weights; document source and training data separately."})
    return {
        "detector": file_reference(resolve_project_file(config.yolo_model)),
        "tracker": tracker_reference(config.tracker),
        "encoder": {"backend": config.encoder_backend, "model_name": config.reid_model_name,
                    "checkpoint": checkpoint},
    }


class RunArtifacts:
    def __init__(self, *, run_id: str, config: PipelineConfig, paths: AppPaths, source: str) -> None:
        self.root = paths.output_dir / "runs" / run_id
        self.root.mkdir(parents=True, exist_ok=False)
        self.manifest_path = self.root / "manifest.json"
        self.predictions_path = self.root / "frames.jsonl"
        self.tracking_path = self.root / "tracking_mot.txt"
        self.processed_frames = 0
        self.metadata: dict[str, Any] = {
            "schema_version": 1, "run_id": run_id, "status": "running", "started_at": utc_now_iso(),
            "configuration": asdict(config), "source": source,
            "input": file_reference(resolve_project_file(source)) if Path(source).is_file() or resolve_project_file(source).is_file()
            else {"label": source, "sha256": None, "note": "Live/non-file source; content cannot be hashed."},
            "database_before": file_reference(paths.db_path),
            "paths": {name: str(value) for name, value in asdict(paths).items()},
            "code": code_reference(), "models": model_references(config),
            "environment": {"python": platform.python_version(), "os": platform.platform(),
                            "machine": platform.machine(), "processor": platform.processor(),
                            "packages": {dist.metadata["Name"]: dist.version
                                         for dist in importlib.metadata.distributions() if dist.metadata["Name"]}},
            "video": {}, "timings": {}, "processed_frames": 0,
            "exports": {"frames_jsonl": str(self.predictions_path), "tracking_mot": str(self.tracking_path),
                        "frame_indices": "1-based", "timestamps": "(frame_index-1)/fps; nominal video time",
                        "tracking_mot_note": "Untracked detections are only in frames.jsonl; no retrospective person-ID backfill."},
        }
        atomic_json(self.manifest_path, self.metadata)
        self._frames = self.predictions_path.open("x", encoding="utf-8")
        try:
            self._tracking = self.tracking_path.open("x", encoding="utf-8")
        except BaseException:
            self._frames.close()
            raise

    def set_video(self, *, fps: float, frame_count: int | None, width: int, height: int, fps_fallback: bool) -> None:
        self.metadata["video"] = {"fps": fps, "declared_frames": frame_count,
                                  "width": width, "height": height, "fps_fallback": fps_fallback,
                                  "timestamp_limit": "Nominal FPS timestamps; variable-frame-rate PTS are not recovered."}
        atomic_json(self.manifest_path, self.metadata)

    def record_frame(self, frame_index: int, fps: float, predictions: list[dict[str, Any]]) -> None:
        if frame_index != self.processed_frames + 1:
            raise ValueError("Frame export must be contiguous and start at 1.")
        self._frames.write(json.dumps({"run_id": self.metadata["run_id"], "frame_index": frame_index,
                                       "timestamp_seconds": (frame_index - 1) / fps,
                                       "detections": predictions}, allow_nan=False) + "\n")
        for prediction in predictions:
            track_id = prediction["track_id"]
            if track_id is not None:
                x1, y1, x2, y2 = prediction["bbox_xyxy"]
                self._tracking.write(f"{frame_index},{track_id},{x1},{y1},{x2-x1},{y2-y1},{prediction['confidence']},-1,-1,-1\n")
        self.processed_frames = frame_index
        self._frames.flush()
        self._tracking.flush()

    def release(self) -> None:
        try:
            self._frames.close()
        finally:
            self._tracking.close()

    def finish(self, *, status: str, error: BaseException | None, processing_seconds: float,
               model_load_seconds: float, total_seconds: float) -> None:
        self.metadata.update(status=status, finished_at=utc_now_iso(), processed_frames=self.processed_frames,
                             error={"type": type(error).__name__, "message": str(error)} if error else None)
        fps = self.metadata["video"].get("fps")
        duration = self.processed_frames / fps if fps else None
        self.metadata["timings"] = {
            "model_load_seconds": model_load_seconds, "processing_seconds": processing_seconds,
            "total_seconds": total_seconds,
            "processed_video_duration_seconds": duration,
            "frames_per_second": self.processed_frames / processing_seconds if processing_seconds > 0 else None,
            "real_time_factor": processing_seconds / duration if duration else None,
            "scope": "Processing includes capture, tracking, encoding, persistence, video/frame export, callbacks and resource closing; excludes model loading and initial artifact/hash setup.",
        }
        self.metadata["exports"]["sha256"] = {
            "frames_jsonl": sha256_file(self.predictions_path), "tracking_mot": sha256_file(self.tracking_path)
        }
        atomic_json(self.manifest_path, self.metadata)
