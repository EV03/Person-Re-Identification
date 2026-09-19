"""Adapter that turns main's frame artifacts into the Details single-person report."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.evaluation.artifacts import atomic_json
from app.evaluation.metrics import compute_single_person_metrics, summary_markdown


STATE_TO_EVENT = {
    "created_identity": "new_person",
    "matched_identity": "matched_person",
    "matched_identity_update_rejected": "matched_person",
    "profile_update": "quality_gated_update",
    "profile_update_rejected": "profile_update_rejected",
    "known_track": "tracking_only",
}


@dataclass(frozen=True)
class SinglePersonReport:
    metrics: dict[str, Any]
    json_path: Path
    markdown_path: Path
    csv_path: Path


def load_prediction_events(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Flatten ``frames.jsonl`` without losing frames or tracker/person separation."""
    events: list[dict[str, Any]] = []
    processed_frames = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            frame = json.loads(line)
            frame_index = int(frame["frame_index"])
            processed_frames = max(processed_frames, frame_index)
            for detection in frame.get("detections", []):
                state = str(detection.get("state") or "unknown")
                event = dict(detection)
                event.update(
                    frame_index=frame_index,
                    event_type=STATE_TO_EVENT.get(state, state),
                    created_new_person=state == "created_identity",
                    score=detection.get("match_score"),
                )
                events.append(event)
    return events, processed_frames


def create_single_person_report(*, predictions_path: Path, output_dir: Path,
                                video_name: str, condition: str = "",
                                expected_person_id: str | None = None,
                                notes: str = "") -> SinglePersonReport:
    events, processed_frames = load_prediction_events(predictions_path)
    created_ids = sorted({str(event["person_id"]) for event in events
                          if event.get("created_new_person") and event.get("person_id")})
    metrics = compute_single_person_metrics(
        events, expected_person_id=expected_person_id,
        processed_frames=processed_frames, expected_real_person_count=1,
        new_person_ids_from_db=created_ids,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "single_person_summary.json"
    markdown_path = output_dir / "single_person_summary.md"
    csv_path = output_dir / "single_person_summary.csv"
    atomic_json(json_path, {
        "method": "details_tracking_single_person_v2",
        "video_name": video_name,
        "condition": condition,
        "notes": notes,
        "predictions_path": str(predictions_path),
        "metrics": metrics,
    })
    markdown_path.write_text(summary_markdown(video_name=video_name, condition=condition,
                                              metrics=metrics, notes=notes), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in metrics.items()})
    return SinglePersonReport(metrics, json_path, markdown_path, csv_path)


def load_main_run_metrics(manifest_path: Path | None) -> dict[str, Any]:
    """Read timing/artifact metrics already produced by main's run manifest."""
    if manifest_path is None or not Path(manifest_path).is_file():
        return {}
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    timings = payload.get("timings") or {}
    return {
        "status": payload.get("status"),
        "processed_frames": payload.get("processed_frames"),
        "processing_seconds": timings.get("processing_seconds"),
        "frames_per_second": timings.get("frames_per_second"),
        "real_time_factor": timings.get("real_time_factor"),
        "frames_jsonl": (payload.get("exports") or {}).get("frames_jsonl"),
        "tracking_mot": (payload.get("exports") or {}).get("tracking_mot"),
    }
