"""SQLite persistence and exact cosine search for person embeddings."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from app.storage.models import MatchResult, PersonRecord, Payload
from app.utils.id_utils import format_person_id, utc_now_iso
from app.utils.image_utils import cosine_similarity, normalize_vector


class SQLiteVectorStore:
    """Small local vector store backed by SQLite.

    It stores one quality-gated mean embedding per synthetic person ID and event
    rows for observations. Similarity search is computed in Python using cosine
    similarity. For a small MVP this keeps setup minimal and avoids a separate
    vector DB server.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Commit or roll back a transaction and always close its connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
        columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        if any(str(column["name"]) == column_name for column in columns):
            return
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    mean_embedding TEXT NOT NULL,
                    observations INTEGER NOT NULL DEFAULT 0,
                    embedding_weight_sum REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    best_snapshot_path TEXT,
                    best_snapshot_quality REAL NOT NULL DEFAULT 0.0
                )
                """
            )
            self._ensure_column(conn, "persons", "embedding_weight_sum", "REAL NOT NULL DEFAULT 1.0")
            self._ensure_column(conn, "persons", "best_snapshot_quality", "REAL NOT NULL DEFAULT 0.0")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    score REAL,
                    bbox_json TEXT NOT NULL,
                    snapshot_path TEXT,
                    created_at TEXT NOT NULL,
                    payload_json TEXT,
                    FOREIGN KEY(person_id) REFERENCES persons(person_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_person_id ON events(person_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_source ON events(source)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    run_id TEXT PRIMARY KEY,
                    mode_id TEXT NOT NULL,
                    mode_name TEXT,
                    pipeline_type TEXT,
                    source TEXT NOT NULL,
                    fps REAL,
                    frame_count INTEGER,
                    width INTEGER,
                    height INTEGER,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analysis_runs_mode_id ON analysis_runs(mode_id)")
            self._ensure_column(conn, "analysis_runs", "status", "TEXT NOT NULL DEFAULT 'unknown'")
            self._ensure_column(conn, "analysis_runs", "processed_frames", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "analysis_runs", "finished_at", "TEXT")
            self._ensure_column(conn, "analysis_runs", "error", "TEXT")


    @staticmethod
    def _vector_to_json(vector: np.ndarray) -> str:
        return json.dumps(normalize_vector(vector).astype(float).tolist())

    @staticmethod
    def _json_to_vector(value: str) -> np.ndarray:
        return np.asarray(json.loads(value), dtype=np.float32)

    @staticmethod
    def _embedding_weight_from_payload(payload: Payload | None) -> float:
        if not payload:
            return 1.0
        raw_weight = payload.get("embedding_weight", payload.get("quality_score", 1.0))
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            return 1.0
        return float(np.clip(weight, 0.05, 1.0))

    @staticmethod
    def _snapshot_quality_from_payload(payload: Payload | None) -> float:
        if not payload:
            return 0.0
        raw_quality = payload.get("snapshot_quality", payload.get("quality_score", 0.0))
        try:
            return float(np.clip(float(raw_quality), 0.0, 1.0))
        except (TypeError, ValueError):
            return 0.0

    def count_persons(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM persons").fetchone()
            return int(row["count"])

    def create_person_id(self) -> str:
        return format_person_id(self.count_persons() + 1)

    def add_analysis_run(
        self,
        run_id: str,
        mode_id: str,
        mode_name: str,
        pipeline_type: str,
        source: str,
        fps: float | None,
        frame_count: int | None,
        width: int | None,
        height: int | None,
        metadata: Payload | None = None,
    ) -> None:
        now = utc_now_iso()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO analysis_runs (
                    run_id, mode_id, mode_name, pipeline_type, source, fps,
                    frame_count, width, height, created_at, metadata_json, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running')
                """,
                (
                    run_id,
                    mode_id,
                    mode_name,
                    pipeline_type,
                    source,
                    fps,
                    frame_count,
                    width,
                    height,
                    now,
                    metadata_json,
                ),
            )

    def finish_analysis_run(self, run_id: str, *, status: str, processed_frames: int, error: str | None) -> None:
        if status not in {"completed", "failed"}:
            raise ValueError("Unknown final run status.")
        with self._connect() as conn:
            conn.execute("UPDATE analysis_runs SET status=?, processed_frames=?, finished_at=?, error=? WHERE run_id=?",
                         (status, processed_frames, utc_now_iso(), error, run_id))

    def search(self, embedding: np.ndarray, threshold: float, exclude_person_ids: set[str] | None = None) -> MatchResult | None:
        exclude_person_ids = exclude_person_ids or set()
        query = normalize_vector(embedding)
        best_person_id: str | None = None
        best_score = -1.0

        with self._connect() as conn:
            rows = conn.execute("SELECT person_id, mean_embedding FROM persons").fetchall()

        for row in rows:
            person_id = str(row["person_id"])
            if person_id in exclude_person_ids:
                continue
            candidate = self._json_to_vector(row["mean_embedding"])
            if candidate.shape != query.shape:
                # A database can contain old ColorHistogram 32D vectors while
                # OSNet emits 512D vectors. Ignore incompatible rows instead of
                # crashing with a shape mismatch.
                continue
            score = cosine_similarity(query, candidate)
            if score > best_score:
                best_person_id = person_id
                best_score = score

        if best_person_id is None or best_score < threshold:
            return None

        return MatchResult(person_id=best_person_id, score=best_score, is_new=False)

    def add_or_update_person(
        self,
        person_id: str,
        embedding: np.ndarray,
        source: str,
        frame_index: int,
        track_id: int,
        bbox_xyxy: tuple[int, int, int, int],
        score: float | None,
        snapshot_path: str | None,
        payload: Payload | None = None,
    ) -> None:
        now = utc_now_iso()
        payload = payload or {}
        embedding = normalize_vector(embedding)
        bbox_json = json.dumps(list(bbox_xyxy))
        payload_json = json.dumps(payload, ensure_ascii=False)
        embedding_weight = self._embedding_weight_from_payload(payload)
        snapshot_quality = self._snapshot_quality_from_payload(payload)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT mean_embedding, observations, embedding_weight_sum, best_snapshot_path, best_snapshot_quality
                FROM persons
                WHERE person_id = ?
                """,
                (person_id,),
            ).fetchone()

            if row is None:
                conn.execute(
                    """
                    INSERT INTO persons (
                        person_id, mean_embedding, observations, embedding_weight_sum,
                        created_at, last_seen, best_snapshot_path, best_snapshot_quality
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        person_id,
                        self._vector_to_json(embedding),
                        1,
                        embedding_weight,
                        now,
                        now,
                        snapshot_path,
                        snapshot_quality,
                    ),
                )
            else:
                old_embedding = self._json_to_vector(row["mean_embedding"])
                old_observations = int(row["observations"])
                old_weight_sum = float(row["embedding_weight_sum"] or max(old_observations, 1))
                new_observations = old_observations + 1
                new_weight_sum = old_weight_sum + embedding_weight

                if old_embedding.shape == embedding.shape:
                    mean_embedding = normalize_vector(
                        (old_embedding * old_weight_sum + embedding * embedding_weight) / new_weight_sum
                    )
                else:
                    # Encoder changed. Replace the incompatible mean vector with
                    # the current vector and continue without breaking the run.
                    mean_embedding = embedding
                    new_weight_sum = embedding_weight

                best_snapshot_path = row["best_snapshot_path"] or snapshot_path
                best_snapshot_quality = float(row["best_snapshot_quality"] or 0.0)
                if snapshot_path and snapshot_quality >= best_snapshot_quality:
                    best_snapshot_path = snapshot_path
                    best_snapshot_quality = snapshot_quality

                conn.execute(
                    """
                    UPDATE persons
                    SET mean_embedding = ?, observations = ?, embedding_weight_sum = ?,
                        last_seen = ?, best_snapshot_path = ?, best_snapshot_quality = ?
                    WHERE person_id = ?
                    """,
                    (
                        self._vector_to_json(mean_embedding),
                        new_observations,
                        new_weight_sum,
                        now,
                        best_snapshot_path,
                        best_snapshot_quality,
                        person_id,
                    ),
                )

            conn.execute(
                """
                INSERT INTO events (person_id, source, frame_index, track_id, score, bbox_json, snapshot_path, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (person_id, source, frame_index, track_id, score, bbox_json, snapshot_path, now, payload_json),
            )

    def list_persons(self) -> list[PersonRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT person_id, observations, created_at, last_seen, best_snapshot_path
                FROM persons
                ORDER BY person_id ASC
                """
            ).fetchall()

        return [
            PersonRecord(
                person_id=str(row["person_id"]),
                observations=int(row["observations"]),
                created_at=str(row["created_at"]),
                last_seen=str(row["last_seen"]),
                best_snapshot_path=row["best_snapshot_path"],
            )
            for row in rows
        ]

    def persons_dataframe(self) -> pd.DataFrame:
        records = self.list_persons()
        return pd.DataFrame([record.__dict__ for record in records])

    def events_dataframe(self, limit: int = 200) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_id, person_id, source, frame_index, track_id, score, bbox_json, snapshot_path, created_at, payload_json
                FROM events
                ORDER BY event_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        records: list[dict[str, object]] = []
        for row in rows:
            record = dict(row)
            payload: dict[str, object] = {}
            if record.get("payload_json"):
                try:
                    payload = json.loads(str(record["payload_json"]))
                except json.JSONDecodeError:
                    payload = {}

            record["event_type"] = payload.get("event_type")
            record["quality_score"] = payload.get("quality_score")
            record["quality_average"] = payload.get("quality_average")
            record["good_frame_count"] = payload.get("good_frame_count")
            records.append(record)

        return pd.DataFrame(records)

    def analysis_runs_dataframe(self, limit: int = 100) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, mode_id, mode_name, pipeline_type, source, fps, frame_count, width, height, created_at,
                       status, processed_frames, finished_at, error
                FROM analysis_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])
