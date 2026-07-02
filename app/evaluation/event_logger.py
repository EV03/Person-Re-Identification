from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class EvaluationEventLogger:
    """Collects pipeline callback events and writes them as CSV/JSONL.

    The logger intentionally stores raw per-detection/per-ReID events. Metrics are
    calculated afterwards so the same run can be re-evaluated with different
    thresholds without rerunning the video.
    """

    def __init__(self, *, output_dir: Path, video_name: str, condition: str = "") -> None:
        self.output_dir = Path(output_dir)
        self.video_name = video_name
        self.condition = condition
        self.events: list[dict[str, Any]] = []

    def log(self, event: dict[str, object]) -> None:
        record: dict[str, Any] = dict(event)
        record.setdefault("video_name", self.video_name)
        record.setdefault("condition", self.condition)
        self.events.append(record)

    @staticmethod
    def _csv_value(value: Any) -> Any:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        return value

    def save(self) -> tuple[Path, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "events.csv"
        jsonl_path = self.output_dir / "events.jsonl"

        keys: list[str] = []
        for event in self.events:
            for key in event.keys():
                if key not in keys:
                    keys.append(key)

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for event in self.events:
                writer.writerow({key: self._csv_value(event.get(key)) for key in keys})

        with jsonl_path.open("w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

        return csv_path, jsonl_path
