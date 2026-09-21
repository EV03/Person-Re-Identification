"""Run one G1-G4 video each with B0, A2 and A3 in isolated units.

The script extracts the four known videos from the supplied ZIP, derives all
three variants from one calibrated baseline preset, runs every video/variant
combination with a fresh profile database and writes machine-readable summaries
plus manual annotation templates for the paper evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from app.config import AppPaths, PipelineConfig  # noqa: E402
from app.evaluation.artifacts import atomic_json, file_reference  # noqa: E402
from app.evaluation.runner import run_unit  # noqa: E402
from app.modes.mode_registry import get_mode, list_modes  # noqa: E402
from app.utils.id_utils import make_run_id, utc_now_iso  # noqa: E402


DEFAULT_ZIP = Path(r"C:\Users\voen0\Downloads\WhatsApp Unknown 2026-09-19 at 16.55.21.zip")


@dataclass(frozen=True)
class GroupVideo:
    group: str
    filename: str
    scenario: str
    expected_event: str


DEFAULT_GROUPS = {
    "G1": GroupVideo(
        "G1",
        "WhatsApp Video 2026-09-19 at 14.39.42.mp4",
        "Freie Sicht und unbekannter Eintritt",
        "unknown_entry",
    ),
    "G2": GroupVideo(
        "G2",
        "WhatsApp Video 2026-09-19 at 14.39.49.mp4",
        "Verlassen und Rückkehr",
        "return",
    ),
    "G3": GroupVideo(
        "G3",
        "WhatsApp Video 2026-09-19 at 14.40.01.mp4",
        "Ähnliche schwarze Kleidung und Rückkehr",
        "return",
    ),
    "G4": GroupVideo(
        "G4",
        "WhatsApp Video 2026-09-19 at 14.39.55.mp4",
        "Kreuzung oder Verdeckung",
        "transition_window",
    ),
}

VARIANT_ORDER = ("B0", "A2", "A3")

AUTOMATIC_SUMMARY_FIELDS = (
    "group",
    "scenario",
    "source_file",
    "variant",
    "mode_id",
    "status",
    "processed_frames",
    "video_duration_seconds",
    "processing_seconds",
    "frames_per_second",
    "real_time_factor",
    "detections",
    "frames_without_detections",
    "unique_tracks",
    "unique_assigned_person_ids",
    "created_identities",
    "matched_identities",
    "accepted_profile_updates",
    "rejected_profile_updates",
    "blocked_by_overlap",
    "blocked_by_overlap_cooldown",
    "below_candidate_quality",
    "below_initial_blur",
    "below_border_blur",
    "below_initial_aspect_ratio",
    "below_update_quality",
    "experiment_manifest",
    "run_manifest",
    "frames_jsonl",
    "annotated_video",
    "error",
)

MANUAL_EVENT_FIELDS = (
    "group",
    "variant",
    "event_id",
    "event_type",
    "gt_person",
    "source_file",
    "registration_person_id",
    "registration_ok",
    "first_visible_frame",
    "decision_deadline_frame",
    "first_person_id_within_window",
    "first_decision_frame",
    "outcome",
    "person_id_at_window_end",
    "decision_time_seconds",
    "transition_frame",
    "profile_person_id",
    "crop_owner_gt",
    "profile_update_accepted",
    "update_classification",
    "visibility_excluded",
    "reviewer",
    "notes",
)

PAPER_METRIC_FIELDS = (
    "group",
    "variant",
    "eligible_returns",
    "correct_returns",
    "false_existing_ids",
    "new_ids_for_registered_persons",
    "no_timely_id",
    "unknown_entries",
    "correct_unknown_entries",
    "registration_failures",
    "evaluable_accepted_updates",
    "false_accepted_updates",
    "evaluable_rejected_updates",
    "correct_rejected_updates",
    "median_seconds_to_correct_id",
    "reviewer",
    "notes",
)


def parse_group_overrides(values: list[str]) -> dict[str, GroupVideo]:
    groups = dict(DEFAULT_GROUPS)
    for value in values:
        if "=" not in value:
            raise ValueError("--group-video erwartet GROUP=DATEINAME, beispielsweise G2=clip.mp4")
        group, filename = (part.strip() for part in value.split("=", 1))
        group = group.upper()
        if group not in groups or not filename:
            raise ValueError(f"Ungültige Gruppenzuordnung: {value}")
        current = groups[group]
        groups[group] = replace(current, filename=filename)
    return groups


def archive_entries(zip_path: Path) -> dict[str, Any]:
    try:
        with ZipFile(zip_path) as archive:
            videos = [entry for entry in archive.infolist() if not entry.is_dir() and entry.filename.lower().endswith(".mp4")]
            by_name: dict[str, Any] = {}
            for entry in videos:
                name = Path(entry.filename).name
                if name in by_name:
                    raise ValueError(f"Das ZIP enthält den Videonamen mehrfach: {name}")
                by_name[name] = entry
            return by_name
    except BadZipFile as exc:
        raise ValueError(f"Ungültiges ZIP-Archiv: {zip_path}") from exc


def validate_sources(zip_path: Path, groups: dict[str, GroupVideo]) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP nicht gefunden: {zip_path}")
    entries = archive_entries(zip_path)
    missing = [item.filename for item in groups.values() if item.filename not in entries]
    if missing:
        raise FileNotFoundError(f"Diese zugeordneten Videos fehlen im ZIP: {', '.join(missing)}")
    if len({item.filename for item in groups.values()}) != len(groups):
        raise ValueError("Jede Versuchsgruppe muss einem anderen Video zugeordnet sein.")


def extract_sources(zip_path: Path, groups: dict[str, GroupVideo], destination: Path) -> dict[str, Path]:
    destination.mkdir(parents=True, exist_ok=False)
    extracted: dict[str, Path] = {}
    with ZipFile(zip_path) as archive:
        entries = {Path(entry.filename).name: entry for entry in archive.infolist() if not entry.is_dir()}
        for group, item in groups.items():
            target = destination / f"{group}_{Path(item.filename).name}"
            with archive.open(entries[item.filename]) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            extracted[group] = target.resolve()
    return extracted


def build_variants(baseline: PipelineConfig, device: str | None) -> dict[str, PipelineConfig]:
    overrides: dict[str, Any] = {"max_frames": 0}
    if device is not None:
        overrides["device"] = device
    b0 = replace(
        baseline,
        **overrides,
        mode_id="eval_b0",
        mode_name=f"B0 - {baseline.mode_name}",
    )
    a2 = replace(
        b0,
        mode_id="eval_a2",
        mode_name=f"A2 - {baseline.mode_name} ohne Qualitätsschwellen",
        min_embedding_quality=0.0,
        min_initial_blur_score=0.0,
        min_border_blur_score=0.0,
        min_initial_aspect_ratio_score=0.0,
        min_update_quality=0.0,
    )
    a3 = replace(
        b0,
        mode_id="eval_a3",
        mode_name=f"A3 - {baseline.mode_name} ohne Update-Schutz",
        min_update_similarity=-1.0,
    )
    return {"B0": b0, "A2": a2, "A3": a3}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_experiment(
    *, group: GroupVideo, variant: str, source: Path, experiment_path: Path
) -> dict[str, Any]:
    experiment = load_json(experiment_path)
    run_entry = experiment["runs"][0]
    run_manifest_path = Path(run_entry["manifest"])
    run = load_json(run_manifest_path)
    summary = run.get("summary", {})
    timings = run.get("timings", {})
    exports = run.get("exports", {})
    row = {field: "" for field in AUTOMATIC_SUMMARY_FIELDS}
    row.update(
        group=group.group,
        scenario=group.scenario,
        source_file=source.name,
        variant=variant,
        mode_id=run.get("configuration", {}).get("mode_id", ""),
        status=run.get("status", ""),
        processed_frames=run.get("processed_frames", ""),
        video_duration_seconds=timings.get("processed_video_duration_seconds", ""),
        processing_seconds=timings.get("processing_seconds", ""),
        frames_per_second=timings.get("frames_per_second", ""),
        real_time_factor=timings.get("real_time_factor", ""),
        experiment_manifest=str(experiment_path.resolve()),
        run_manifest=str(run_manifest_path.resolve()),
        frames_jsonl=exports.get("frames_jsonl", ""),
        annotated_video=exports.get("annotated_video", ""),
    )
    for field in AUTOMATIC_SUMMARY_FIELDS:
        if field in summary:
            row[field] = summary[field]
    return row


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_manual_templates(
    batch_root: Path,
    groups: dict[str, GroupVideo],
    variants: tuple[str, ...],
) -> None:
    event_rows = []
    metric_rows = []
    for group in groups.values():
        for variant in variants:
            event_rows.append(
                {
                    "group": group.group,
                    "variant": variant,
                    "event_id": f"{group.group}_E01",
                    "event_type": group.expected_event,
                    "source_file": group.filename,
                }
            )
            metric_rows.append({"group": group.group, "variant": variant})
    write_csv(batch_root / "manual_event_log.csv", MANUAL_EVENT_FIELDS, event_rows)
    write_csv(batch_root / "paper_metrics_template.csv", PAPER_METRIC_FIELDS, metric_rows)


def markdown_value(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_paper_summary(path: Path, rows: list[dict[str, Any]], baseline_mode: str) -> None:
    lines = [
        "# Automatische Laufzusammenfassung",
        "",
        f"Baseline-Preset: `{baseline_mode}`. ReID-Erfolgsraten bleiben bis zur manuellen GT-Prüfung offen.",
        "",
        "| Gruppe | Variante | Frames | IDs erzeugt | Re-IDs | Updates angenommen/abgelehnt | FPS | RTF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {group} | {variant} | {frames} | {created} | {matched} | {accepted}/{rejected} | {fps} | {rtf} |".format(
                group=row["group"],
                variant=row["variant"],
                frames=markdown_value(row["processed_frames"]),
                created=markdown_value(row["created_identities"]),
                matched=markdown_value(row["matched_identities"]),
                accepted=markdown_value(row["accepted_profile_updates"]),
                rejected=markdown_value(row["rejected_profile_updates"]),
                fps=markdown_value(row["frames_per_second"]),
                rtf=markdown_value(row["real_time_factor"]),
            )
        )
    lines.extend(
        [
            "",
            "Die Zähler sind Systemausgaben und keine Ground Truth. Für das Paper müssen `manual_event_log.csv` und anschließend `paper_metrics_template.csv` ausgefüllt werden.",
            "Eine Zeile in `manual_event_log.csv` ist nur ein Starter. Für jede reale Person und jedes Rückkehr-, Eintritts- oder Updateereignis sind zusätzliche Zeilen anzulegen.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="ZIP mit den vier MP4-Dateien")
    argument_parser.add_argument(
        "--baseline-mode",
        default="best-calibrated",
        help="Kalibriertes Ausgangspreset für B0; A2 und A3 werden exakt daraus abgeleitet",
    )
    argument_parser.add_argument("--device", default=None, help="Optional: cpu, cuda, cuda:0 oder auto")
    argument_parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "experiments",
        help="Übergeordnetes Ausgabeverzeichnis",
    )
    argument_parser.add_argument(
        "--groups",
        nargs="+",
        choices=tuple(DEFAULT_GROUPS),
        default=list(DEFAULT_GROUPS),
        help="Optional nur bestimmte Gruppen ausführen",
    )
    argument_parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANT_ORDER,
        default=list(VARIANT_ORDER),
        help="Optional nur bestimmte Varianten ausführen",
    )
    argument_parser.add_argument(
        "--group-video",
        action="append",
        default=[],
        metavar="GROUP=DATEINAME",
        help="Standardzuordnung eines Videos überschreiben; mehrfach verwendbar",
    )
    argument_parser.add_argument("--dry-run", action="store_true", help="ZIP, Zuordnung und Preset prüfen, ohne Modelle zu starten")
    return argument_parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    groups = parse_group_overrides(args.group_video)
    groups = {group: groups[group] for group in args.groups}
    zip_path = args.zip.expanduser().resolve()
    validate_sources(zip_path, groups)

    app_paths = AppPaths()
    modes = list_modes(app_paths)
    if args.baseline_mode not in modes:
        available = ", ".join(sorted(modes))
        raise KeyError(f"Baseline-Preset '{args.baseline_mode}' fehlt. Verfügbar: {available}")
    baseline = get_mode(args.baseline_mode, app_paths).to_pipeline_config()
    all_variants = build_variants(baseline, args.device)
    variants = {variant: all_variants[variant] for variant in args.variants}

    print(f"ZIP: {zip_path}")
    print(f"Baseline: {args.baseline_mode} (Match={baseline.match_threshold}, Update={baseline.min_update_similarity})")
    for group in groups.values():
        print(f"{group.group}: {group.filename} -> {group.scenario}")
    print(f"Varianten: {', '.join(variants)} ({len(groups) * len(variants)} isolierte Läufe)")
    if args.dry_run:
        print("Dry-run erfolgreich; keine Dateien extrahiert und keine Modelle gestartet.")
        return 0

    batch_root = args.root.expanduser().resolve() / f"four_groups_{make_run_id()}"
    batch_root.mkdir(parents=True, exist_ok=False)
    extracted = extract_sources(zip_path, groups, batch_root / "input")
    write_manual_templates(batch_root, groups, tuple(variants))

    batch_manifest_path = batch_root / "batch_manifest.json"
    batch_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now_iso(),
        "source_zip": file_reference(zip_path),
        "baseline_preset": args.baseline_mode,
        "group_mapping": {group: asdict(item) for group, item in groups.items()},
        "variants": {variant: asdict(config) for variant, config in variants.items()},
        "units": [],
    }
    atomic_json(batch_manifest_path, batch_manifest)

    rows: list[dict[str, Any]] = []
    failed = False
    try:
        for group_name, group in groups.items():
            for variant, config in variants.items():
                print(f"Starte {group_name}/{variant}: {extracted[group_name].name}", flush=True)
                unit_entry = {"group": group_name, "variant": variant, "status": "running"}
                batch_manifest["units"].append(unit_entry)
                atomic_json(batch_manifest_path, batch_manifest)
                try:
                    experiment = run_unit(
                        [extracted[group_name]],
                        config,
                        root=batch_root / group_name,
                        base_paths=app_paths,
                    )
                    row = row_from_experiment(
                        group=group,
                        variant=variant,
                        source=extracted[group_name],
                        experiment_path=experiment,
                    )
                    rows.append(row)
                    unit_entry.update(status="completed", experiment_manifest=str(experiment.resolve()))
                    print(f"Fertig {group_name}/{variant}: {experiment}", flush=True)
                except Exception as exc:
                    failed = True
                    unit_entry.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
                    row = {field: "" for field in AUTOMATIC_SUMMARY_FIELDS}
                    row.update(
                        group=group_name,
                        scenario=group.scenario,
                        source_file=extracted[group_name].name,
                        variant=variant,
                        mode_id=config.mode_id,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    rows.append(row)
                    print(f"FEHLER {group_name}/{variant}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                finally:
                    write_csv(batch_root / "automatic_run_summary.csv", AUTOMATIC_SUMMARY_FIELDS, rows)
                    write_paper_summary(batch_root / "automatic_paper_summary.md", rows, args.baseline_mode)
                    atomic_json(batch_manifest_path, batch_manifest)
        batch_manifest["status"] = "failed" if failed else "completed"
        return 1 if failed else 0
    except BaseException as exc:
        batch_manifest.update(
            status="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    finally:
        batch_manifest["finished_at"] = utc_now_iso()
        atomic_json(batch_manifest_path, batch_manifest)
        print(f"Auswertungspaket: {batch_root}")


if __name__ == "__main__":
    raise SystemExit(main())
