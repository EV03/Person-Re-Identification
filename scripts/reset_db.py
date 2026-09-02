from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppPaths


RESET_ROOT = (PROJECT_ROOT / "data").resolve()


def safe_reset_target(path: Path, allowed_root: Path = RESET_ROOT) -> Path:
    """Resolve and validate a path before destructive reset operations."""
    resolved_root = allowed_root.resolve()
    resolved_target = path.resolve()
    if resolved_target == resolved_root or resolved_root not in resolved_target.parents:
        raise ValueError(
            f"Refusing to delete unsafe reset target '{resolved_target}'. "
            f"Targets must be children of '{resolved_root}'."
        )
    return resolved_target


def remove_path(path: Path) -> None:
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main() -> None:
    paths = AppPaths()
    reset_targets = [
        paths.db_path,
        paths.db_path.with_suffix(paths.db_path.suffix + "-wal"),
        paths.db_path.with_suffix(paths.db_path.suffix + "-shm"),
        paths.snapshot_dir,
        paths.output_dir,
    ]
    validated_targets = [safe_reset_target(path) for path in reset_targets]
    for target in validated_targets:
        remove_path(target)
    paths.ensure()
    print("Reset complete.")


if __name__ == "__main__":
    main()
