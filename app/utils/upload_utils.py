from __future__ import annotations

import hashlib
import re
from pathlib import Path
from uuid import uuid4


ALLOWED_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4"}
MAX_UPLOAD_BYTES = 2 * 1024**3


def sanitize_uploaded_filename(filename: str) -> tuple[str, str]:
    """Return a safe filename stem and validated lowercase video suffix."""
    basename = str(filename).replace("\\", "/").rsplit("/", 1)[-1].strip()
    path = Path(basename)
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        supported = ", ".join(sorted(ALLOWED_VIDEO_SUFFIXES))
        raise ValueError(f"Unsupported video type '{suffix or 'none'}'. Supported types: {supported}")

    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", path.stem)
    stem = stem.strip("._-")[:80] or "upload"
    return stem, suffix


def persist_uploaded_video(
    input_dir: Path,
    filename: str,
    data: bytes | bytearray | memoryview,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Path:
    """Store an uploaded video once using a deterministic content-based path."""
    payload = memoryview(data)
    if payload.nbytes > max_bytes:
        limit_mb = max_bytes / (1024**2)
        raise ValueError(f"Uploaded video exceeds the {limit_mb:.0f} MB size limit.")

    stem, suffix = sanitize_uploaded_filename(filename)
    digest = hashlib.sha256(payload).hexdigest()[:16]
    resolved_input_dir = input_dir.resolve()
    resolved_input_dir.mkdir(parents=True, exist_ok=True)
    target = (resolved_input_dir / f"{stem}_{digest}{suffix}").resolve()
    if resolved_input_dir not in target.parents:
        raise ValueError("Resolved upload path is outside the configured input directory.")
    if target.exists():
        return target

    temporary = resolved_input_dir / f".{digest}.{uuid4().hex}.upload"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
