"""A fresh experiment unit per variant/repetition, shared profiles per unit.

Multiple sources in ONE unit form a related registration/return sequence.
Each source gets a fresh pipeline/tracker while the unit's database is retained.
Independent scenarios must be dispatched as separate units.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from app.config import AppPaths, PipelineConfig, PROJECT_ROOT
from app.evaluation.artifacts import atomic_json, file_reference
from app.pipeline.orchestrator import PersonReIdPipeline
from app.storage.encoder_paths import paths_for_encoder
from app.utils.id_utils import make_run_id, safe_source_name, utc_now_iso


def create_unit_paths(*, root: Path | None = None, base_paths: AppPaths | None = None,
                      mode_id: str = "experiment") -> AppPaths:
    base = base_paths or AppPaths()
    root = Path(root) if root is not None else base.output_dir.parent / "experiments"
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    unit = root / f"{safe_source_name(mode_id)}_{make_run_id()}"
    # Exclusive creation is important: never reuse a possibly populated DB.
    unit.mkdir(parents=True, exist_ok=False)
    return AppPaths(db_path=unit / "db" / "reid.sqlite3", snapshot_dir=unit / "snapshots",
                    output_dir=unit / "output", mode_dir=base.mode_dir, input_dir=base.input_dir)


def run_unit(sources: list[Path], config: PipelineConfig, *, root: Path | None = None,
             base_paths: AppPaths | None = None,
             pipeline_factory: Callable = PersonReIdPipeline) -> Path:
    if not sources:
        raise ValueError("An experiment unit needs at least one source.")
    sources = [Path(source).resolve() for source in sources]
    if any(not source.is_file() for source in sources):
        raise FileNotFoundError("Every experiment source must be an existing video file.")
    paths = create_unit_paths(root=root, base_paths=base_paths, mode_id=config.mode_id)
    manifest_path = paths.output_dir.parent / "experiment.json"
    paths = paths_for_encoder(paths, config)
    experiment_root = manifest_path.parent.resolve()
    manifest = {
        "schema_version": 2, "status": "running", "started_at": utc_now_iso(),
        "configuration": asdict(config), "sources": [file_reference(source) for source in sources],
        "database": str(paths.db_path), "database_initial_state": "new, empty",
        "database_relative_to_experiment": paths.db_path.resolve().relative_to(experiment_root).as_posix(),
        "profile_policy": "Shared across sources in this unit only; fresh tracker per source.",
        "runs": [],
    }
    atomic_json(manifest_path, manifest)
    try:
        for source in sources:
            # Do not cache/reuse the tracker or pipeline between sequences.
            entry = {"source": str(source), "status": "running"}
            manifest["runs"].append(entry)
            atomic_json(manifest_path, manifest)
            pipeline = None
            try:
                pipeline = pipeline_factory(config=config, paths=paths)
                result = pipeline.process(str(source))
                entry.update(status="completed", run_id=result.run_id, manifest=str(result.manifest_path),
                             predictions=str(result.predictions_path), processed_frames=result.processed_frames,
                             manifest_relative_to_experiment=result.manifest_path.resolve().relative_to(experiment_root).as_posix(),
                             predictions_relative_to_experiment=result.predictions_path.resolve().relative_to(experiment_root).as_posix())
            except BaseException:
                entry.update(status="failed", manifest=str(getattr(pipeline, "last_manifest_path", "unavailable")))
                raise
            atomic_json(manifest_path, manifest)
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        manifest["finished_at"] = utc_now_iso()
        if paths.db_path.is_file():
            manifest["database_after"] = file_reference(paths.db_path)
            manifest["database_after"]["relative_to_experiment"] = paths.db_path.resolve().relative_to(experiment_root).as_posix()
        atomic_json(manifest_path, manifest)
    return manifest_path
