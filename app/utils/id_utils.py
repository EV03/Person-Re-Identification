from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def format_person_id(number: int) -> str:
    return f"person_{number:06d}"


def safe_source_name(source: object) -> str:
    raw = str(source)
    for char in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|", " ", "[", "]"]:
        raw = raw.replace(char, "_")
    return raw[-80:] if len(raw) > 80 else raw


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1
