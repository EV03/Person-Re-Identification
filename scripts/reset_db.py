from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppPaths
from app.evaluation.db_snapshot import backup_sqlite_database, remove_sqlite_files


def remove_path(path: Path) -> None:
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Back up and reset the local ReID database safely.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required confirmation. Without it, the script only explains what would be reset.",
    )
    parser.add_argument(
        "--delete-outputs",
        action="store_true",
        help="Also back up and remove annotated output videos. Outputs are preserved by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = AppPaths()
    if not args.confirm:
        print(f"Database: {paths.db_path}")
        print(f"Snapshots: {paths.snapshot_dir}")
        print(f"Outputs remain untouched: {paths.output_dir}")
        print("No files changed. Run again with --confirm to create a backup and reset the database.")
        return

    backup_root = paths.output_dir.parent / "backups" / f"reset_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    backup_root.mkdir(parents=True, exist_ok=False)

    if paths.db_path.exists():
        backup_sqlite_database(paths.db_path, backup_root / "reid.sqlite3")
    if paths.snapshot_dir.exists():
        shutil.copytree(paths.snapshot_dir, backup_root / "snapshots")
    if args.delete_outputs and paths.output_dir.exists():
        shutil.copytree(paths.output_dir, backup_root / "output")

    remove_sqlite_files(paths.db_path)
    remove_path(paths.snapshot_dir)
    if args.delete_outputs:
        remove_path(paths.output_dir)
    paths.ensure()
    print(f"Reset complete. Backup: {backup_root}")
    print(f"Annotated outputs preserved: {not args.delete_outputs}")


if __name__ == "__main__":
    main()
