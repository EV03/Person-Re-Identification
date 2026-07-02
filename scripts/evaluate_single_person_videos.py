from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppPaths, PipelineConfig  # noqa: E402
from app.evaluation.db_snapshot import backup_sqlite_database, copy_sqlite_files, remove_sqlite_files  # noqa: E402
from app.evaluation.event_logger import EvaluationEventLogger  # noqa: E402
from app.evaluation.manifest import EvaluationVideo, load_manifest, write_manifest_template  # noqa: E402
from app.evaluation.metrics import compute_single_person_metrics, summary_markdown  # noqa: E402
from app.pipeline.orchestrator import PersonReIdPipeline  # noqa: E402
from app.utils.id_utils import safe_source_name  # noqa: E402


def _now_run_id() -> str:
    return datetime.now().strftime("eval_%Y%m%d_%H%M%S")


def _project_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _normalize_test_mode(value: str) -> str:
    aliases = {
        "adaptive": "adaptive_calibration",
        "calibration_then_adaptive": "adaptive_calibration",
        "growing_calibration": "adaptive_calibration",
        "calibration_then_grow": "adaptive_calibration",
    }
    return aliases.get(str(value or "").strip().lower(), str(value or "").strip().lower())


def _build_config(args: argparse.Namespace, *, calibration: bool = False) -> PipelineConfig:
    config = PipelineConfig()
    config.max_frames = int(args.max_frames)
    config.device = str(args.device)
    config.match_threshold = float(args.match_threshold)
    config.strong_match_threshold = float(args.strong_match_threshold)
    config.weak_match_threshold = float(args.weak_match_threshold)
    config.new_person_max_score = float(args.new_person_max_score)
    config.new_person_min_evidence_events = int(args.new_person_min_evidence_events)
    config.new_person_evidence_window_frames = int(args.new_person_evidence_window_frames)
    config.new_person_low_match_ratio = float(args.new_person_low_match_ratio)
    config.motion_identity_bonus = float(args.motion_identity_bonus)
    config.motion_identity_max_frame_gap = int(args.motion_identity_max_frame_gap)
    config.motion_identity_max_distance_fraction = float(args.motion_identity_max_distance_fraction)
    config.merge_candidate_threshold = float(args.merge_candidate_threshold)
    config.merge_candidate_min_events = int(args.merge_candidate_min_events)
    config.detection_confidence = float(args.detection_confidence)
    config.image_size = int(args.image_size)
    config.reid_every_n_frames = int(args.reid_every_n_frames)
    config.min_good_frames_before_reid = int(args.min_good_frames_before_reid)
    config.min_embedding_quality = float(args.min_embedding_quality)
    config.min_update_quality = float(args.min_update_quality)
    config.min_crop_height = int(args.min_crop_height)
    config.min_crop_width = int(args.min_crop_width)
    config.crop_padding = float(args.crop_padding)
    config.enable_detail_analysis = not bool(args.disable_detail_analysis)
    config.detail_weight = float(args.detail_weight)
    config.detail_min_confidence = float(args.detail_min_confidence)
    if calibration:
        config.detection_confidence = float(args.calibration_detection_confidence)
        config.image_size = int(args.calibration_image_size)
        config.reid_every_n_frames = int(args.calibration_reid_every_n_frames)
        config.min_good_frames_before_reid = int(args.calibration_min_good_frames_before_reid)
        config.min_embedding_quality = float(args.calibration_min_embedding_quality)
        config.min_update_quality = float(args.calibration_min_update_quality)
        config.min_crop_height = int(args.calibration_min_crop_height)
        config.min_crop_width = int(args.calibration_min_crop_width)
        config.crop_padding = float(args.calibration_crop_padding)
        if bool(args.calibration_disable_detail_analysis):
            config.enable_detail_analysis = False
            config.detail_weight = 0.0
    config.draw_debug = not bool(args.no_output_video)
    config.live_preview_every_n_frames = 999999
    return config


def _paths_for_run(db_path: Path, run_dir: Path) -> AppPaths:
    return AppPaths(
        db_path=db_path,
        snapshot_dir=run_dir / "snapshots",
        output_dir=run_dir / "output_videos",
        mode_dir=PROJECT_ROOT / "data" / "modes",
        input_dir=PROJECT_ROOT / "data" / "input",
    )


def _read_person_ids(db_path: Path) -> set[str]:
    import sqlite3

    if not db_path.exists():
        return set()
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT person_id FROM persons").fetchall()
    except sqlite3.Error:
        return set()
    finally:
        conn.close()
    return {str(row[0]) for row in rows}


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for record in records:
        for key in record.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in record.items()
                }
            )


def _progress_callback(frame: int, total: int | None, message: str) -> None:
    if frame == 1 or frame % 100 == 0:
        total_text = "?" if total is None else str(total)
        print(f"  Frame {frame}/{total_text}: {message}")


def _run_one_video(
    *,
    row: EvaluationVideo,
    db_path: Path,
    run_dir: Path,
    config: PipelineConfig,
    expected_person_id: str | None,
) -> tuple[dict[str, Any], str | None]:
    video_path = _project_path(row.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_slug = safe_source_name(f"{video_path.stem}_{row.condition}_{row.phase}")
    video_dir = run_dir / "videos" / video_slug
    video_dir.mkdir(parents=True, exist_ok=True)

    persons_before = _read_person_ids(db_path)
    logger = EvaluationEventLogger(output_dir=video_dir, video_name=video_path.name, condition=row.condition)

    paths = _paths_for_run(db_path, run_dir)
    pipeline = PersonReIdPipeline(config=config, paths=paths)
    try:
        print(f"\n▶ {row.phase.upper()}: {video_path.name} [{row.condition}]")
        result = pipeline.process(
            str(video_path),
            progress_callback=_progress_callback,
            frame_callback=None,
            event_callback=logger.log,
        )
    finally:
        pipeline.close()

    persons_after = _read_person_ids(db_path)
    new_person_ids = sorted(persons_after - persons_before)

    logger.save()
    metrics = compute_single_person_metrics(
        logger.events,
        expected_person_id=expected_person_id or row.expected_person_id,
        processed_frames=result.processed_frames,
        expected_real_person_count=1,
        new_person_ids_from_db=new_person_ids,
    )

    metrics_payload = {
        "video_name": video_path.name,
        "video_path": str(video_path),
        "phase": row.phase,
        "condition": row.condition,
        "notes": row.notes,
        "pipeline_result": {
            "run_id": result.run_id,
            "processed_frames": result.processed_frames,
            "created_persons": result.created_persons,
            "matched_events": result.matched_events,
            "warnings": result.warnings,
            "output_video_path": str(result.output_video_path) if result.output_video_path else None,
        },
        "metrics": metrics,
    }
    _save_json(video_dir / "summary.json", metrics_payload)
    (video_dir / "summary.md").write_text(
        summary_markdown(video_name=video_path.name, condition=row.condition, metrics=metrics, notes=row.notes),
        encoding="utf-8",
    )

    print(
        "  Ergebnis: "
        f"tracking={metrics['tracking_rating']} | "
        f"reid={metrics['reid_rating']} | "
        f"dominant_person={metrics.get('dominant_person_id')} | "
        f"expected_ratio={metrics.get('expected_person_ratio')}"
    )

    detected_expected = expected_person_id or row.expected_person_id or metrics.get("dominant_person_id")
    return metrics_payload, str(detected_expected) if detected_expected else None


def command_init_manifest(args: argparse.Namespace) -> None:
    video_root = _project_path(args.video_root)
    output_path = _project_path(args.output)
    path = write_manifest_template(video_root, output_path)
    print(f"Manifest erstellt: {path}")
    print("Danach bei einem Kalibrier-Video die Spalte phase auf 'calibration' setzen.")


def command_run(args: argparse.Namespace) -> None:
    manifest_path = _project_path(args.manifest)
    rows = load_manifest(manifest_path)
    if not rows:
        raise RuntimeError("Manifest contains no videos.")

    run_id = args.run_id or _now_run_id()
    test_mode = _normalize_test_mode(args.test_mode)
    output_root = _project_path(args.output_dir)
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    working_db = run_dir / "working" / "reid_eval.sqlite3"
    calibrated_db = run_dir / "calibration" / "calibrated_reid.sqlite3"
    config = _build_config(args, calibration=False)
    calibration_config = _build_config(args, calibration=True)
    expected_person_id = args.expected_person_id or None

    run_config = {
        "run_id": run_id,
        "manifest": str(manifest_path),
        "test_mode": _normalize_test_mode(args.test_mode),
        "expected_person_id_start": expected_person_id,
        "config": {
            "max_frames": config.max_frames,
            "device": config.device,
            "match_threshold": config.match_threshold,
            "strong_match_threshold": config.strong_match_threshold,
            "weak_match_threshold": config.weak_match_threshold,
            "new_person_max_score": config.new_person_max_score,
            "new_person_min_evidence_events": config.new_person_min_evidence_events,
            "new_person_evidence_window_frames": config.new_person_evidence_window_frames,
            "new_person_low_match_ratio": config.new_person_low_match_ratio,
            "detection_confidence": config.detection_confidence,
            "image_size": config.image_size,
            "min_embedding_quality": config.min_embedding_quality,
            "min_update_quality": config.min_update_quality,
            "min_crop_height": config.min_crop_height,
            "min_crop_width": config.min_crop_width,
            "crop_padding": config.crop_padding,
            "calibration_image_size": calibration_config.image_size,
            "calibration_min_embedding_quality": calibration_config.min_embedding_quality,
            "calibration_min_update_quality": calibration_config.min_update_quality,
            "reid_every_n_frames": config.reid_every_n_frames,
            "min_good_frames_before_reid": config.min_good_frames_before_reid,
            "enable_detail_analysis": config.enable_detail_analysis,
            "detail_weight": config.detail_weight,
            "detail_min_confidence": config.detail_min_confidence,
        },
    }
    _save_json(run_dir / "run_config.json", run_config)

    calibration_rows = [row for row in rows if row.phase == "calibration"]
    test_rows = [row for row in rows if row.phase == "test"]
    summary_records: list[dict[str, Any]] = []

    remove_sqlite_files(working_db)

    if args.base_db_snapshot:
        base_db = _project_path(args.base_db_snapshot)
        copy_sqlite_files(base_db, working_db)
        backup_sqlite_database(working_db, calibrated_db)
        print(f"Basis-DB geladen: {base_db}")
    elif calibration_rows:
        for row in calibration_rows:
            payload, detected = _run_one_video(
                row=row,
                db_path=working_db,
                run_dir=run_dir,
                config=calibration_config,
                expected_person_id=expected_person_id or row.expected_person_id,
            )
            metrics = payload["metrics"]
            summary_records.append(
                {
                    "video_name": payload["video_name"],
                    "phase": payload["phase"],
                    "condition": payload["condition"],
                    **{key: metrics.get(key) for key in _SUMMARY_KEYS},
                }
            )
            if not expected_person_id and detected:
                expected_person_id = detected
        backup_sqlite_database(working_db, calibrated_db)
        print(f"\nKalibrierte DB gespeichert: {calibrated_db}")
    else:
        print("Keine Kalibrierung und keine Basis-DB angegeben. Tests starten mit leerer DB.")
        backup_empty_parent = working_db.parent
        backup_empty_parent.mkdir(parents=True, exist_ok=True)

    if not expected_person_id and args.expected_person_id:
        expected_person_id = args.expected_person_id

    calibration_state = {
        "expected_person_id": expected_person_id,
        "calibrated_db": str(calibrated_db) if calibrated_db.exists() else None,
        "calibration_videos": [asdict(row) for row in calibration_rows],
    }
    _save_json(run_dir / "calibration_state.json", calibration_state)

    if test_mode == "adaptive_calibration" and not calibrated_db.exists():
        print("Adaptive Kalibrierung ohne Kalibrier-DB: Lauf startet wie learn_through mit leerer DB.")

    if test_mode == "fixed_db":
        for row in test_rows:
            video_db = run_dir / "video_databases" / f"{safe_source_name(row.video_path.stem)}.sqlite3"
            if calibrated_db.exists():
                copy_sqlite_files(calibrated_db, video_db)
            else:
                remove_sqlite_files(video_db)
            payload, _ = _run_one_video(
                row=row,
                db_path=video_db,
                run_dir=run_dir,
                config=config,
                expected_person_id=expected_person_id or row.expected_person_id,
            )
            metrics = payload["metrics"]
            summary_records.append(
                {
                    "video_name": payload["video_name"],
                    "phase": payload["phase"],
                    "condition": payload["condition"],
                    **{key: metrics.get(key) for key in _SUMMARY_KEYS},
                }
            )
    else:
        mode_dir_name = "adaptive_calibration" if test_mode == "adaptive_calibration" else "learn_through"
        learn_db = run_dir / mode_dir_name / "reid_eval.sqlite3"
        if calibrated_db.exists():
            copy_sqlite_files(calibrated_db, learn_db)
        else:
            remove_sqlite_files(learn_db)
        for row in test_rows:
            payload, _ = _run_one_video(
                row=row,
                db_path=learn_db,
                run_dir=run_dir,
                config=config,
                expected_person_id=expected_person_id or row.expected_person_id,
            )
            metrics = payload["metrics"]
            summary_records.append(
                {
                    "video_name": payload["video_name"],
                    "phase": payload["phase"],
                    "condition": payload["condition"],
                    **{key: metrics.get(key) for key in _SUMMARY_KEYS},
                }
            )
        if learn_db.exists():
            backup_sqlite_database(learn_db, run_dir / mode_dir_name / "final_adaptive_reid.sqlite3")
            print(f"\nFinale wachsende DB gespeichert: {run_dir / mode_dir_name / 'final_adaptive_reid.sqlite3'}")

    _write_summary_csv(run_dir / "all_videos_summary.csv", summary_records)
    print(f"\nFertig. Ergebnisordner: {run_dir}")
    print(f"Gesamtübersicht: {run_dir / 'all_videos_summary.csv'}")


_SUMMARY_KEYS = [
    "expected_person_id",
    "processed_frames",
    "event_count",
    "frames_with_events",
    "unique_track_ids",
    "dominant_track_id",
    "dominant_track_ratio",
    "track_switch_count",
    "tracking_rating",
    "unique_person_ids",
    "dominant_person_id",
    "dominant_person_ratio",
    "expected_person_ratio",
    "primary_person_id",
    "primary_person_ratio",
    "adaptive_reid_rating",
    "profile_growth_event_count",
    "profile_fragmentation_index",
    "pending_weak_match_count",
    "pending_new_person_count",
    "strong_match_count",
    "weak_match_count",
    "low_match_count",
    "person_switch_count",
    "new_person_count_from_db",
    "reid_rating",
    "fragmentation_rating",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-person evaluation for the Person ReID MVP.")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init-manifest", help="Scan a folder and create a manifest CSV template.")
    init_parser.add_argument("--video-root", required=True, help="Folder that contains test videos.")
    init_parser.add_argument("--output", default="data/evaluation_manifest.csv", help="Output CSV path.")
    init_parser.set_defaults(func=command_init_manifest)

    run_parser = sub.add_parser("run", help="Run evaluation based on a manifest CSV.")
    run_parser.add_argument("--manifest", required=True, help="CSV with phase,video_path,condition,expected_person_id,notes.")
    run_parser.add_argument("--output-dir", default="data/evaluation_runs", help="Output root for evaluation runs.")
    run_parser.add_argument("--run-id", default="", help="Optional fixed run id/folder name.")
    run_parser.add_argument("--base-db-snapshot", default="", help="Optional calibrated SQLite DB to use instead of calibration rows.")
    run_parser.add_argument("--expected-person-id", default="", help="Expected global person_id, e.g. person_000001.")
    run_parser.add_argument(
        "--test-mode",
        choices=["fixed_db", "learn_through", "adaptive_calibration", "adaptive", "calibration_then_adaptive", "growing_calibration", "calibration_then_grow"],
        default="fixed_db",
        help=(
            "fixed_db: jedes Testvideo startet von derselben kalibrierten DB-Kopie. "
            "learn_through: Videos laufen nacheinander und die DB wächst. "
            "adaptive_calibration: Kalibrierung als Startprofil, danach kontrolliert wachsender Praxistest."
        ),
    )
    run_parser.add_argument("--max-frames", type=int, default=0, help="0 means full video.")
    run_parser.add_argument("--device", default="auto", help="auto, cpu or cuda.")
    run_parser.add_argument("--match-threshold", type=float, default=0.74)
    run_parser.add_argument("--strong-match-threshold", type=float, default=0.82)
    run_parser.add_argument("--weak-match-threshold", type=float, default=0.68)
    run_parser.add_argument("--new-person-max-score", type=float, default=0.58)
    run_parser.add_argument("--new-person-min-evidence-events", type=int, default=6)
    run_parser.add_argument("--new-person-evidence-window-frames", type=int, default=30)
    run_parser.add_argument("--new-person-low-match-ratio", type=float, default=0.80)
    run_parser.add_argument("--motion-identity-bonus", type=float, default=0.04)
    run_parser.add_argument("--motion-identity-max-frame-gap", type=int, default=15)
    run_parser.add_argument("--motion-identity-max-distance-fraction", type=float, default=0.15)
    run_parser.add_argument("--merge-candidate-threshold", type=float, default=0.86)
    run_parser.add_argument("--merge-candidate-min-events", type=int, default=8)
    run_parser.add_argument("--detection-confidence", type=float, default=0.40)
    run_parser.add_argument("--image-size", type=int, default=960)
    run_parser.add_argument("--min-embedding-quality", type=float, default=0.60)
    run_parser.add_argument("--min-update-quality", type=float, default=0.75)
    run_parser.add_argument("--min-crop-height", type=int, default=120)
    run_parser.add_argument("--min-crop-width", type=int, default=45)
    run_parser.add_argument("--crop-padding", type=float, default=0.08)
    run_parser.add_argument("--calibration-detection-confidence", type=float, default=0.45)
    run_parser.add_argument("--calibration-image-size", type=int, default=1280)
    run_parser.add_argument("--calibration-reid-every-n-frames", type=int, default=3)
    run_parser.add_argument("--calibration-min-good-frames-before-reid", type=int, default=5)
    run_parser.add_argument("--calibration-min-embedding-quality", type=float, default=0.70)
    run_parser.add_argument("--calibration-min-update-quality", type=float, default=0.80)
    run_parser.add_argument("--calibration-min-crop-height", type=int, default=160)
    run_parser.add_argument("--calibration-min-crop-width", type=int, default=60)
    run_parser.add_argument("--calibration-crop-padding", type=float, default=0.10)
    run_parser.add_argument("--calibration-disable-detail-analysis", action="store_true", default=True)
    run_parser.add_argument("--reid-every-n-frames", type=int, default=5)
    run_parser.add_argument("--min-good-frames-before-reid", type=int, default=3)
    run_parser.add_argument("--disable-detail-analysis", action="store_true")
    run_parser.add_argument("--detail-weight", type=float, default=0.05)
    run_parser.add_argument("--detail-min-confidence", type=float, default=0.70)
    run_parser.add_argument("--no-output-video", action="store_true", help="Do not draw debug overlays; still processes the video.")
    run_parser.set_defaults(func=command_run)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
