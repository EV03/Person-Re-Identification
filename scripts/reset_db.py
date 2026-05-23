from __future__ import annotations

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppPaths


def remove_path(path: Path) -> None:
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main() -> None:
    paths = AppPaths()
    for db_file in [paths.db_path, paths.db_path.with_suffix(paths.db_path.suffix + "-wal"), paths.db_path.with_suffix(paths.db_path.suffix + "-shm")]:
        remove_path(db_file)
    remove_path(paths.snapshot_dir)
    remove_path(paths.output_dir)
    paths.ensure()
    print("Reset complete.")


if __name__ == "__main__":
    main()
