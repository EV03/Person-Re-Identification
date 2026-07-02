from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


@dataclass(frozen=True)
class EvaluationVideo:
    video_path: Path
    phase: str
    condition: str
    expected_person_id: str | None = None
    notes: str = ""


def load_manifest(path: Path) -> list[EvaluationVideo]:
    path = Path(path)
    rows: list[EvaluationVideo] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            video_path = Path((raw.get("video_path") or "").strip())
            if not video_path:
                continue
            phase = (raw.get("phase") or "test").strip().lower()
            if phase in {"calibration", "calibrate", "kalibrierung"}:
                phase = "calibration"
            elif phase in {"test", "evaluation", "eval"}:
                phase = "test"
            else:
                raise ValueError(f"Unknown phase '{phase}' for video {video_path}")
            rows.append(
                EvaluationVideo(
                    video_path=video_path,
                    phase=phase,
                    condition=(raw.get("condition") or video_path.parent.name).strip(),
                    expected_person_id=(raw.get("expected_person_id") or "").strip() or None,
                    notes=(raw.get("notes") or "").strip(),
                )
            )
    return rows


def write_manifest_template(video_root: Path, output_path: Path) -> Path:
    video_root = Path(video_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        path for path in video_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "video_path", "condition", "expected_person_id", "notes"])
        writer.writeheader()
        for video in videos:
            writer.writerow(
                {
                    "phase": "test",
                    "video_path": str(video),
                    "condition": video.parent.name,
                    "expected_person_id": "",
                    "notes": "",
                }
            )
    return output_path
