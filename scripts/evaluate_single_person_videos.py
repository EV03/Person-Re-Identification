"""Single-person evaluator for any preset using main's artifact contract.

Examples:
    python scripts/evaluate_single_person_videos.py init-manifest --video-root Test-daten
    python scripts/evaluate_single_person_videos.py run --manifest data/evaluation_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppPaths  # noqa: E402
from app.evaluation.single_person import create_single_person_report  # noqa: E402
from app.modes.mode_registry import get_mode  # noqa: E402
from app.pipeline.orchestrator import PersonReIdPipeline  # noqa: E402
from app.storage.encoder_paths import paths_for_encoder  # noqa: E402
from app.utils.id_utils import safe_source_name  # noqa: E402


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def init_manifest(video_root: Path, output: Path) -> Path:
    extensions = {".mp4", ".mov", ".avi", ".mkv"}
    rows = []
    for path in sorted(video_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            rows.append({
                "phase": "test", "video_path": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
                "condition": path.parent.name, "expected_person_id": "", "notes": "",
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "video_path", "condition", "expected_person_id", "notes"])
        writer.writeheader()
        writer.writerows(rows)
    return output


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [{key: str(value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]
    for row in rows:
        if row.get("phase") not in {"calibration", "test"}:
            raise ValueError("Manifest phase must be 'calibration' or 'test'.")
        if not row.get("video_path"):
            raise ValueError("Every manifest row needs video_path.")
    return rows


def copy_database(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if source.exists():
        shutil.copy2(source, destination)


def run_video(row: dict[str, str], *, paths: AppPaths, config,
              expected_person_id: str | None) -> tuple[dict[str, object], str | None]:
    video = project_path(row["video_path"])
    if not video.is_file():
        raise FileNotFoundError(video)
    pipeline = PersonReIdPipeline(config=config, paths=paths)
    result = pipeline.process(str(video))
    if result.predictions_path is None:
        raise RuntimeError("Pipeline produced no frames.jsonl artifact.")
    report_dir = result.predictions_path.parent
    report = create_single_person_report(
        predictions_path=result.predictions_path, output_dir=report_dir,
        video_name=video.name, condition=row.get("condition", ""),
        expected_person_id=expected_person_id or row.get("expected_person_id") or None,
        notes=row.get("notes", ""),
    )
    detected = expected_person_id or row.get("expected_person_id") or report.metrics.get("dominant_person_id")
    return {
        "video_name": video.name, "phase": row["phase"], "condition": row.get("condition", ""),
        **report.metrics, "report": str(report.json_path), "run_manifest": str(result.manifest_path),
    }, str(detected) if detected else None


def run_manifest(args: argparse.Namespace) -> Path:
    rows = load_manifest(project_path(args.manifest))
    run_id = args.run_id or datetime.now().strftime("single_%Y%m%d_%H%M%S")
    run_dir = project_path(args.output_dir) / safe_source_name(run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    mode = get_mode(args.preset)
    config = mode.to_pipeline_config(max_frames=args.max_frames, device=args.device)
    base_paths = AppPaths(
        db_path=run_dir / "working" / "reid.sqlite3", snapshot_dir=run_dir / "snapshots",
        output_dir=run_dir / "output", mode_dir=PROJECT_ROOT / "data" / "modes",
        input_dir=PROJECT_ROOT / "data" / "input",
    )
    expected_id = args.expected_person_id or None
    summaries: list[dict[str, object]] = []

    calibration_rows = [row for row in rows if row["phase"] == "calibration"]
    test_rows = [row for row in rows if row["phase"] == "test"]
    for row in calibration_rows:
        summary, detected = run_video(row, paths=base_paths, config=config,
                                      expected_person_id=expected_id)
        summaries.append(summary)
        expected_id = expected_id or detected

    working_db = paths_for_encoder(base_paths, config).db_path
    calibrated_db = run_dir / "calibration" / "calibrated_reid.sqlite3"
    if working_db.exists():
        copy_database(working_db, calibrated_db)

    if args.test_mode == "fixed_db":
        for index, row in enumerate(test_rows, start=1):
            test_paths = replace(base_paths, db_path=run_dir / "fixed" / f"video_{index}" / "reid.sqlite3",
                                 snapshot_dir=run_dir / "fixed" / f"video_{index}" / "snapshots",
                                 output_dir=run_dir / "fixed" / f"video_{index}" / "output")
            target_db = paths_for_encoder(test_paths, config).db_path
            if calibrated_db.exists():
                copy_database(calibrated_db, target_db)
            summary, _ = run_video(row, paths=test_paths, config=config,
                                   expected_person_id=expected_id)
            summaries.append(summary)
    else:
        adaptive_paths = replace(base_paths, db_path=run_dir / args.test_mode / "reid.sqlite3",
                                 snapshot_dir=run_dir / args.test_mode / "snapshots",
                                 output_dir=run_dir / args.test_mode / "output")
        target_db = paths_for_encoder(adaptive_paths, config).db_path
        if calibrated_db.exists():
            copy_database(calibrated_db, target_db)
        for row in test_rows:
            summary, _ = run_video(row, paths=adaptive_paths, config=config,
                                   expected_person_id=expected_id)
            summaries.append(summary)

    summary_path = run_dir / "all_videos_summary.csv"
    keys = list(dict.fromkeys(key for summary in summaries for key in summary))
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(summaries)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Controlled single-person evaluation for any ReID preset.")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-manifest")
    init.add_argument("--video-root", required=True)
    init.add_argument("--output", default="data/evaluation_manifest.csv")
    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--output-dir", default="data/evaluation_runs")
    run.add_argument("--run-id", default="")
    run.add_argument("--preset", default="default")
    run.add_argument("--test-mode", choices=["fixed_db", "learn_through", "adaptive_calibration"], default="fixed_db")
    run.add_argument("--expected-person-id", default="")
    run.add_argument("--max-frames", type=int, default=0)
    run.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "init-manifest":
        print(init_manifest(project_path(args.video_root), project_path(args.output)))
    else:
        print(run_manifest(args))


if __name__ == "__main__":
    main()
