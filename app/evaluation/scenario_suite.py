"""G1-G4 scenario discovery, multi-preset execution and comparison reports."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable

from app.config import AppPaths, PROJECT_ROOT
from app.evaluation.artifacts import atomic_json
from app.evaluation.metrics import compute_scenario_diagnostics
from app.evaluation.runner import run_unit
from app.evaluation.single_person import load_main_run_metrics, load_prediction_events
from app.modes.mode_registry import get_mode
from app.utils.id_utils import make_run_id, safe_source_name


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
SCENARIO_DESCRIPTIONS = {
    "G1": "Zwei Personen ohne Kreuzung",
    "G2": "Personen verlassen das Bild und kehren zurück",
    "G3": "Personen mit ähnlicher Kleidung",
    "G4": "Personen kreuzen sich",
}
DEFAULT_SUITE_MODES = (
    "default",
    "colorhist",
    "no_quality_thresholds",
    "no_update_similarity",
    "details_tracking",
    "details_no_reranking",
    "details_no_weak_zone",
    "details_immediate_new_person",
    "details_no_overlap_protection",
    "details_no_motion_bonus",
)
MANIFEST_FIELDS = ("group", "sequence", "video_path", "expected_person_count", "notes")


@dataclass(frozen=True)
class ScenarioVideo:
    group: str
    sequence: str
    video_path: Path
    expected_person_count: int = 2
    notes: str = ""


def _portable_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def discover_scenario_videos(video_root: Path) -> list[ScenarioVideo]:
    """Discover ``G1``...``G4`` folders; subfolders define shared sequences."""
    video_root = Path(video_root).resolve()
    if not video_root.is_dir():
        raise FileNotFoundError(video_root)
    directories = {path.name.upper(): path for path in video_root.iterdir() if path.is_dir()}
    entries: list[ScenarioVideo] = []
    for group, description in SCENARIO_DESCRIPTIONS.items():
        group_dir = directories.get(group)
        if group_dir is None:
            continue
        for video in sorted(path for path in group_dir.rglob("*")
                            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS):
            relative = video.relative_to(group_dir)
            sequence = (relative.parent.as_posix()
                        if relative.parent != Path(".") else video.stem)
            entries.append(ScenarioVideo(group, sequence, video, 2, description))
    return entries


def write_scenario_manifest(entries: Iterable[ScenarioVideo], output: Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow({
                "group": entry.group,
                "sequence": entry.sequence,
                "video_path": _portable_path(entry.video_path),
                "expected_person_count": entry.expected_person_count,
                "notes": entry.notes,
            })
    return output


def load_scenario_manifest(path: Path) -> list[ScenarioVideo]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    entries: list[ScenarioVideo] = []
    for row in rows:
        group = str(row.get("group") or "").strip().upper()
        if group not in SCENARIO_DESCRIPTIONS:
            raise ValueError(f"Unknown scenario group: {group or '<empty>'}")
        sequence = str(row.get("sequence") or "").strip()
        if not sequence:
            raise ValueError("Every scenario row needs a sequence.")
        video_path = resolve_project_path(str(row.get("video_path") or "").strip()).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        expected = int(row.get("expected_person_count") or 2)
        if expected < 1:
            raise ValueError("expected_person_count must be at least 1.")
        entries.append(ScenarioVideo(group, sequence, video_path, expected,
                                     str(row.get("notes") or "").strip()))
    if not entries:
        raise ValueError("The G1-G4 manifest contains no videos.")
    return entries


def _flatten_report_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items()
            if key not in {"main_metrics", "identity_diagnostics"}}
    flat.update({f"main_{key}": value for key, value in row.get("main_metrics", {}).items()})
    flat.update({f"identity_{key}": value for key, value in row.get("identity_diagnostics", {}).items()})
    return {
        key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
        for key, value in flat.items()
    }


def _average(rows: list[dict[str, Any]], section: str, key: str) -> float | None:
    values = [row.get(section, {}).get(key) for row in rows if row.get("status") == "completed"]
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and value is not None]
    return fmean(numeric) if numeric else None


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["mode_id"]), str(row["group"]))].append(row)
    result = []
    for (mode_id, group), items in groups.items():
        result.append({
            "mode_id": mode_id,
            "mode_name": items[0].get("mode_name"),
            "group": group,
            "runs": len(items),
            "completed": sum(item.get("status") == "completed" for item in items),
            "mean_fps": _average(items, "main_metrics", "frames_per_second"),
            "mean_real_time_factor": _average(items, "main_metrics", "real_time_factor"),
            "mean_person_assignment_ratio": _average(items, "identity_diagnostics", "person_assignment_ratio"),
            "mean_profile_count_delta": _average(items, "identity_diagnostics", "profile_count_delta"),
            "mean_track_to_person_output_switches": _average(
                items, "identity_diagnostics", "track_to_person_output_switches"),
            "mean_person_to_track_fragment_surplus": _average(
                items, "identity_diagnostics", "person_to_track_fragment_surplus"),
        })
    return sorted(result, key=lambda item: (item["mode_id"], item["group"]))


def _fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _markdown_report(rows: list[dict[str, Any]], aggregates: list[dict[str, Any]], *, title: str) -> str:
    lines = [
        f"# {title}", "",
        "Die Laufzeit-/Artefaktmetriken stammen aus der Main-Pipeline. Die Identitätsdiagnostik wird aus `frames.jsonl` für jede Pipeline gleich berechnet.",
        "Ohne dichte Ground-Truth-Boxen und reale Identitätslabels sind die Switch-/Fragmentierungswerte Output-Diagnosen, keine IDF1-, MOTA- oder echten ID-Switch-Metriken.",
        "",
        "## Aggregat nach Preset und Gruppe", "",
        "| Preset | Gruppe | Läufe | OK | FPS | RTF | Zuordnung | Profil-Delta | Track→Person-Wechsel | Person→Track-Fragmente |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        lines.append(
            f"| {item['mode_id']} | {item['group']} | {item['runs']} | {item['completed']} | "
            f"{_fmt(item['mean_fps'], 2)} | {_fmt(item['mean_real_time_factor'])} | "
            f"{_fmt(item['mean_person_assignment_ratio'])} | {_fmt(item['mean_profile_count_delta'])} | "
            f"{_fmt(item['mean_track_to_person_output_switches'])} | "
            f"{_fmt(item['mean_person_to_track_fragment_surplus'])} |"
        )
    lines.extend(["", "## Einzelläufe", "",
                  "| Preset | Gruppe | Sequenz | Video | Status | Frames | Unique Tracks | Unique Profile | Profil-Delta | Strong/Weak/Low |",
                  "|---|---|---|---|---|---:|---:|---:|---:|---:|"])
    for row in rows:
        diag = row.get("identity_diagnostics", {})
        main = row.get("main_metrics", {})
        zones = f"{diag.get('strong_match_count', 0)}/{diag.get('weak_match_count', 0)}/{diag.get('low_match_count', 0)}"
        lines.append(
            f"| {row['mode_id']} | {row['group']} | {row['sequence']} | {row.get('video_name', '-')} | "
            f"{row['status']} | {main.get('processed_frames', 0) or 0} | {diag.get('unique_track_ids', 0)} | "
            f"{diag.get('unique_person_ids', 0)} | {diag.get('profile_count_delta', 'n/a')} | {zones} |"
        )
    return "\n".join(lines) + "\n"


def write_suite_reports(rows: list[dict[str, Any]], output_dir: Path,
                        *, title: str = "G1-G4 Pipelinevergleich") -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_rows(rows)
    json_path = output_dir / "comparison.json"
    csv_path = output_dir / "comparison.csv"
    aggregate_csv_path = output_dir / "aggregate.csv"
    markdown_path = output_dir / "report.md"
    atomic_json(json_path, {"title": title, "rows": rows, "aggregates": aggregates})
    flat_rows = [_flatten_report_row(row) for row in rows]
    if flat_rows:
        fields = list(dict.fromkeys(key for row in flat_rows for key in row))
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat_rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    if aggregates:
        with aggregate_csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
            writer.writeheader()
            writer.writerows(aggregates)
    else:
        aggregate_csv_path.write_text("", encoding="utf-8")
    markdown_path.write_text(_markdown_report(rows, aggregates, title=title), encoding="utf-8")

    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(exist_ok=True)
    for mode_id in sorted({str(row["mode_id"]) for row in rows}):
        mode_rows = [row for row in rows if row["mode_id"] == mode_id]
        mode_aggregates = [item for item in aggregates if item["mode_id"] == mode_id]
        profile_name = safe_source_name(mode_id)
        (profile_dir / f"{profile_name}.md").write_text(
            _markdown_report(mode_rows, mode_aggregates, title=f"G1-G4 – {mode_id}"), encoding="utf-8")
        atomic_json(profile_dir / f"{profile_name}.json", {
            "mode_id": mode_id, "rows": mode_rows, "aggregates": mode_aggregates,
        })
        mode_flat_rows = [_flatten_report_row(row) for row in mode_rows]
        fields = list(dict.fromkeys(key for row in mode_flat_rows for key in row))
        with (profile_dir / f"{profile_name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(mode_flat_rows)
    return {"json": json_path, "csv": csv_path, "aggregate_csv": aggregate_csv_path,
            "markdown": markdown_path, "profiles": profile_dir}


def run_scenario_suite(
    entries: list[ScenarioVideo], *, mode_ids: Iterable[str] = DEFAULT_SUITE_MODES,
    output_root: Path, base_paths: AppPaths | None = None, repetitions: int = 1,
    checkpoint: str | None = None, device: str | None = None, max_frames: int = 0,
    run_unit_fn: Callable[..., Path] = run_unit,
) -> dict[str, Path]:
    if not entries:
        raise ValueError("At least one G1-G4 video is required.")
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1.")
    base_paths = base_paths or AppPaths()
    suite_dir = Path(output_root) / f"g1_g4_{make_run_id()}"
    suite_dir.mkdir(parents=True, exist_ok=False)
    grouped: dict[tuple[str, str], list[ScenarioVideo]] = defaultdict(list)
    for entry in entries:
        grouped[(entry.group, entry.sequence)].append(entry)
    rows: list[dict[str, Any]] = []
    for mode_id in mode_ids:
        mode = get_mode(mode_id, base_paths)
        overrides: dict[str, Any] = {"max_frames": int(max_frames)}
        if checkpoint is not None:
            overrides["reid_checkpoint"] = checkpoint
        if device is not None:
            overrides["device"] = device
        config = mode.to_pipeline_config(**overrides)
        for (group, sequence), scenario_entries in sorted(grouped.items()):
            scenario_entries = sorted(scenario_entries, key=lambda item: str(item.video_path).lower())
            for repetition in range(1, repetitions + 1):
                try:
                    experiment_path = run_unit_fn(
                        [entry.video_path for entry in scenario_entries], config,
                        root=suite_dir / "units", base_paths=base_paths,
                    )
                    experiment = json.loads(Path(experiment_path).read_text(encoding="utf-8"))
                    experiment_runs = experiment.get("runs", [])
                    if len(experiment_runs) != len(scenario_entries):
                        raise ValueError("Experiment manifest does not contain one run per scenario video.")
                    for entry, run in zip(scenario_entries, experiment_runs):
                        manifest_path = Path(run["manifest"])
                        predictions_path = Path(run["predictions"])
                        events, processed_frames = load_prediction_events(predictions_path)
                        rows.append({
                            "group": group, "group_description": SCENARIO_DESCRIPTIONS[group],
                            "sequence": sequence, "video_name": entry.video_path.name,
                            "source": str(entry.video_path), "expected_person_count": entry.expected_person_count,
                            "notes": entry.notes, "mode_id": mode.mode_id, "mode_name": mode.name,
                            "repetition": repetition, "status": run.get("status", "completed"),
                            "experiment_manifest": str(experiment_path), "run_manifest": str(manifest_path),
                            "predictions": str(predictions_path), "error": None,
                            "main_metrics": load_main_run_metrics(manifest_path),
                            "identity_diagnostics": compute_scenario_diagnostics(
                                events, processed_frames=processed_frames,
                                expected_real_person_count=entry.expected_person_count),
                        })
                except Exception as exc:
                    rows.append({
                        "group": group, "group_description": SCENARIO_DESCRIPTIONS[group],
                        "sequence": sequence, "video_name": " / ".join(
                            entry.video_path.name for entry in scenario_entries),
                        "source": None, "expected_person_count": scenario_entries[0].expected_person_count,
                        "notes": scenario_entries[0].notes, "mode_id": mode.mode_id, "mode_name": mode.name,
                        "repetition": repetition, "status": "failed", "experiment_manifest": None,
                        "run_manifest": None, "predictions": None,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "main_metrics": {}, "identity_diagnostics": {},
                    })
                write_suite_reports(rows, suite_dir)
    return write_suite_reports(rows, suite_dir)
