"""Evaluate the manually referenced events of one four-group experiment batch."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = PROJECT_ROOT / "docs" / "evaluation" / "four_group_event_reference.json"
VARIANTS = ("B0", "A2", "A3")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_frames(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def detections_for_track(
    frames: list[dict[str, Any]], track_id: int, start: int = 1, end: int | None = None
) -> list[tuple[int, dict[str, Any]]]:
    found = []
    for frame in frames:
        frame_index = int(frame["frame_index"])
        if frame_index < start or (end is not None and frame_index > end):
            continue
        found.extend(
            (frame_index, detection)
            for detection in frame["detections"]
            if detection.get("track_id") == track_id
        )
    return found


def first_person_id(detections: list[tuple[int, dict[str, Any]]]) -> tuple[int | None, str | None, str | None]:
    for frame_index, detection in detections:
        person_id = detection.get("person_id")
        if person_id is not None:
            return frame_index, str(person_id), str(detection.get("state"))
    return None, None, None


def last_person_id(detections: list[tuple[int, dict[str, Any]]]) -> str | None:
    assigned = [str(detection["person_id"]) for _, detection in detections if detection.get("person_id")]
    return assigned[-1] if assigned else None


def evaluate_event(
    event: dict[str, Any], variant: str, row: dict[str, str], frames: list[dict[str, Any]], fps: float,
    decision_window_seconds: float,
) -> dict[str, Any]:
    first_frame = int(event["first_visible_frame"])
    deadline = max(
        frame["frame_index"]
        for frame in frames
        if (frame["frame_index"] - first_frame) / fps <= decision_window_seconds
    )
    observed = detections_for_track(frames, int(event["track_id"]), first_frame, deadline)
    decision_frame, assigned_id, decision_state = first_person_id(observed)
    original_id = None
    if event["kind"] != "unknown_entry":
        _, original_id, _ = first_person_id(
            detections_for_track(frames, int(event["initial_track_id"]), 1, first_frame - 1)
        )
        if original_id is None:
            raise ValueError(f"Keine ursprüngliche Personen-ID für {event['event_id']} in {variant}")

    initial_ids = {
        person_id
        for candidate in event.get("initial_track_ids", [])
        for _, person_id, _ in [first_person_id(detections_for_track(frames, int(candidate), 1, first_frame - 1))]
        if person_id
    }
    if event["kind"] == "unknown_entry":
        outcome = "correct_new_id" if assigned_id and decision_state == "created_identity" else "no_timely_id"
    elif assigned_id is None:
        outcome = "no_timely_id"
    elif assigned_id == original_id:
        outcome = "correct_old_id"
    elif assigned_id in initial_ids:
        outcome = "false_existing_id"
    else:
        outcome = "new_id_for_registered_person"

    return {
        "group": event["group"],
        "variant": variant,
        "event_id": event["event_id"],
        "event_type": event["kind"],
        "gt_person": event["gt_person"],
        "source_file": row["source_file"],
        "track_id": event["track_id"],
        "initial_track_id": event.get("initial_track_id", ""),
        "original_person_id": original_id or "",
        "first_visible_frame": first_frame,
        "decision_deadline_frame": deadline,
        "first_person_id_within_window": assigned_id or "",
        "first_decision_frame": decision_frame or "",
        "decision_state": decision_state or "",
        "outcome": outcome,
        "person_id_at_window_end": last_person_id(observed) or "",
        "decision_time_seconds": ((decision_frame - first_frame) / fps if decision_frame is not None else ""),
        "notes": event.get("notes", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Keine Zeilen für {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ratio_text(numerator: int, denominator: int) -> str:
    return "n. d." if denominator == 0 else f"{numerator}/{denominator} ({100*numerator/denominator:.1f} %)"


def evaluate(batch: Path, reference_path: Path) -> dict[str, Any]:
    reference = read_json(reference_path)
    with (batch / "automatic_run_summary.csv").open(encoding="utf-8-sig") as handle:
        run_rows = list(csv.DictReader(handle))
    runs = {(row["group"], row["variant"]): row for row in run_rows}
    event_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []

    frame_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    fps_cache: dict[tuple[str, str], float] = {}
    for key, row in runs.items():
        manifest = read_json(Path(row["run_manifest"]))
        frame_cache[key] = read_frames(Path(row["frames_jsonl"]))
        fps_cache[key] = float(manifest["video"]["fps"])

    for event in reference["events"]:
        for variant in VARIANTS:
            key = (event["group"], variant)
            event_rows.append(
                evaluate_event(
                    event,
                    variant,
                    runs[key],
                    frame_cache[key],
                    fps_cache[key],
                    float(reference["decision_window_seconds"]),
                )
            )

    for window in reference["update_windows"]:
        for variant in VARIANTS:
            key = (window["group"], variant)
            frames = frame_cache[key]
            attempts = [
                (frame_index, detection)
                for frame_index, detection in detections_for_track(
                    frames, int(window["track_id"]), int(window["start_frame"]), int(window["end_frame"])
                )
                if detection.get("state") in {"profile_update", "profile_update_rejected"}
            ]
            for frame_index, detection in attempts:
                accepted = detection["state"] == "profile_update"
                same_owner = window["profile_owner_gt"] == window["crop_owner_gt"]
                update_rows.append(
                    {
                        "window_id": window["window_id"],
                        "group": window["group"],
                        "variant": variant,
                        "frame_index": frame_index,
                        "track_id": window["track_id"],
                        "profile_owner_gt": window["profile_owner_gt"],
                        "crop_owner_gt": window["crop_owner_gt"],
                        "same_owner": same_owner,
                        "accepted": accepted,
                        "update_similarity": detection.get("update_similarity", ""),
                        "classification": (
                            "correct_accepted_own" if accepted and same_owner
                            else "correct_rejected_foreign" if not accepted and not same_owner
                            else "false_accepted_foreign" if accepted
                            else "correct_crop_rejected"
                        ),
                    }
                )

    metrics: dict[str, Any] = {"variants": {}, "groups": {}, "excluded_events": reference["excluded_events"]}
    for variant in VARIANTS:
        variant_events = [row for row in event_rows if row["variant"] == variant]
        returns = [row for row in variant_events if row["event_type"] == "return"]
        unknown = [row for row in variant_events if row["event_type"] == "unknown_entry"]
        transitions = [row for row in variant_events if row["event_type"] == "transition"]
        successful_return_times = [
            float(row["decision_time_seconds"])
            for row in returns
            if row["outcome"] == "correct_old_id"
        ]
        variant_updates = [row for row in update_rows if row["variant"] == variant]
        own = [row for row in variant_updates if row["same_owner"]]
        foreign = [row for row in variant_updates if not row["same_owner"]]
        run_variant = [row for row in run_rows if row["variant"] == variant]
        total_frames = sum(int(row["processed_frames"]) for row in run_variant)
        total_processing = sum(float(row["processing_seconds"]) for row in run_variant)
        total_video = sum(float(row["video_duration_seconds"]) for row in run_variant)
        metrics["variants"][variant] = {
            "returns_correct": sum(row["outcome"] == "correct_old_id" for row in returns),
            "returns_total": len(returns),
            "false_existing_ids": sum(row["outcome"] == "false_existing_id" for row in returns),
            "new_ids_for_registered_persons": sum(row["outcome"] == "new_id_for_registered_person" for row in returns),
            "no_timely_return_id": sum(row["outcome"] == "no_timely_id" for row in returns),
            "unknown_entries_correct": sum(row["outcome"] == "correct_new_id" for row in unknown),
            "unknown_entries_total": len(unknown),
            "transitions_correct": sum(row["outcome"] == "correct_old_id" for row in transitions),
            "transitions_total": len(transitions),
            "median_seconds_to_correct_return": statistics.median(successful_return_times),
            "own_updates_accepted": sum(row["accepted"] for row in own),
            "own_update_candidates": len(own),
            "foreign_updates_accepted": sum(row["accepted"] for row in foreign),
            "foreign_update_candidates": len(foreign),
            "processed_frames": total_frames,
            "processing_seconds": total_processing,
            "frames_per_second": total_frames / total_processing,
            "real_time_factor": total_processing / total_video,
        }
        for group in ("G2", "G3"):
            group_returns = [row for row in returns if row["group"] == group]
            metrics["groups"].setdefault(group, {})[variant] = {
                "correct": sum(row["outcome"] == "correct_old_id" for row in group_returns),
                "total": len(group_returns),
            }

    write_csv(batch / "evaluated_event_log.csv", event_rows)
    write_csv(batch / "evaluated_update_log.csv", update_rows)
    (batch / "evaluation_results.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Ausgewertete Ergebnisse für das Paper",
        "",
        "Die Zuordnung der realen Personen wurde anhand der Originalvideos, annotierten Videos und Frameprotokolle manuell geprüft.",
        "",
        "| Kenngröße | B0 | A2 | A3 |",
        "|---|---:|---:|---:|",
    ]
    for label, numerator, denominator in (
        ("Rückkehr korrekt", "returns_correct", "returns_total"),
        ("G1: unbekannter Eintritt rechtzeitig", "unknown_entries_correct", "unknown_entries_total"),
        ("G4: Neuzuweisung nach Kreuzung rechtzeitig korrekt", "transitions_correct", "transitions_total"),
        ("Korrekte Eigenupdates im G4-Fenster angenommen", "own_updates_accepted", "own_update_candidates"),
        ("Fremdupdates im G4-Fenster angenommen", "foreign_updates_accepted", "foreign_update_candidates"),
    ):
        values = [ratio_text(metrics["variants"][variant][numerator], metrics["variants"][variant][denominator]) for variant in VARIANTS]
        lines.append(f"| {label} | {' | '.join(values)} |")
    lines.append(
        "| Median bis zur korrekten Rückkehr-ID | "
        + " | ".join(f"{metrics['variants'][variant]['median_seconds_to_correct_return']:.2f} s" for variant in VARIANTS)
        + " |"
    )
    lines.append(
        "| Verarbeitung | "
        + " | ".join(
            f"{metrics['variants'][variant]['frames_per_second']:.2f} FPS / RTF {metrics['variants'][variant]['real_time_factor']:.3f}"
            for variant in VARIANTS
        )
        + " |"
    )
    lines.extend(["", "## Rückkehr nach Gruppe", "", "| Gruppe | B0 | A2 | A3 |", "|---|---:|---:|---:|"])
    for group in ("G2", "G3"):
        values = [ratio_text(metrics["groups"][group][variant]["correct"], metrics["groups"][group][variant]["total"]) for variant in VARIANTS]
        lines.append(f"| {group} | {' | '.join(values)} |")
    lines.extend(
        [
            "",
            "Ein weiterer Eintritt in G2 wurde ausgeschlossen, weil das Video vor Ablauf des vollständigen Zwei-Sekunden-Fensters endet.",
            "Im G4-Fenster traten keine auswertbaren Fremdkandidaten auf; eine Schutzwirkung gegen falsche Updates ist damit dort nicht nachgewiesen.",
            "",
        ]
    )
    (batch / "paper_evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = parser.parse_args()
    metrics = evaluate(args.batch.resolve(), args.reference.resolve())
    print(args.batch.resolve() / "paper_evaluation.md")
    print(json.dumps(metrics["variants"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
