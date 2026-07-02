from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path


SQLITE_SIDE_SUFFIXES = ("-wal", "-shm")


def remove_sqlite_files(db_path: Path) -> None:
    db_path = Path(db_path)
    for candidate in [db_path, *[Path(str(db_path) + suffix) for suffix in SQLITE_SIDE_SUFFIXES]]:
        if candidate.exists():
            candidate.unlink()


def copy_sqlite_files(source_db: Path, target_db: Path) -> None:
    source_db = Path(source_db)
    target_db = Path(target_db)
    target_db.parent.mkdir(parents=True, exist_ok=True)
    remove_sqlite_files(target_db)
    if not source_db.exists():
        raise FileNotFoundError(f"SQLite database does not exist: {source_db}")
    shutil.copy2(source_db, target_db)
    for suffix in SQLITE_SIDE_SUFFIXES:
        source_side = Path(str(source_db) + suffix)
        if source_side.exists():
            shutil.copy2(source_side, Path(str(target_db) + suffix))


def backup_sqlite_database(source_db: Path, target_db: Path) -> None:
    """Create a consistent SQLite backup into a single target .sqlite3 file."""

    source_db = Path(source_db)
    target_db = Path(target_db)
    target_db.parent.mkdir(parents=True, exist_ok=True)
    remove_sqlite_files(target_db)
    if not source_db.exists():
        raise FileNotFoundError(f"SQLite database does not exist: {source_db}")

    source_conn = sqlite3.connect(source_db)
    try:
        target_conn = sqlite3.connect(target_db)
        try:
            source_conn.backup(target_conn)
        finally:
            target_conn.close()
    finally:
        source_conn.close()
