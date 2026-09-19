"""SQLite persistence adapter; identity decisions live in app.reid."""

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
from app.reid.repository import IdentityProfile, ProfileObservation
from app.reid.service import ProfileService


class SQLiteVectorStore:
    """Small local vector store backed by SQLite.

    Persists profiles, observations and run metadata. Matching and profile
    arithmetic belong to ProfileService and its replaceable policies.
    Compatibility methods delegate to that service; new callers should inject
    ProfileService explicitly.
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
            self._ensure_column(conn, "persons", "embedding_sum", "TEXT")
            self._ensure_column(conn, "persons", "best_snapshot_quality", "REAL NOT NULL DEFAULT 0.0")
            self._ensure_column(conn, "persons", "detail_vector", "TEXT")
            self._ensure_column(conn, "persons", "detail_weight_sum", "REAL NOT NULL DEFAULT 0.0")

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
        return json.dumps(vector.astype(float).tolist())

    @staticmethod
    def _json_to_vector(value: str) -> np.ndarray:
        return np.asarray(json.loads(value), dtype=np.float64)

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

    @classmethod
    def _row_to_profile(cls, row: sqlite3.Row) -> IdentityProfile:
        return IdentityProfile(
            person_id=str(row["person_id"]), embedding=cls._json_to_vector(row["mean_embedding"]),
            observations=int(row["observations"]),
            embedding_weight_sum=float(row["embedding_weight_sum"] or max(int(row["observations"]), 1)),
            created_at=str(row["created_at"]), last_seen=str(row["last_seen"]),
            best_snapshot_path=row["best_snapshot_path"],
            best_snapshot_quality=float(row["best_snapshot_quality"] or 0.0),
            embedding_sum=cls._json_to_vector(row["embedding_sum"]) if row["embedding_sum"] is not None else None,
            detail_vector=cls._json_to_vector(row["detail_vector"]) if row["detail_vector"] is not None else None,
            detail_weight_sum=float(row["detail_weight_sum"] or 0.0),
        )

    def get_profile(self, person_id: str) -> IdentityProfile | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM persons WHERE person_id = ?", (person_id,)).fetchone()
        return self._row_to_profile(row) if row is not None else None

    def iter_profiles(self) -> list[IdentityProfile]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM persons").fetchall()
        return [self._row_to_profile(row) for row in rows]

    def save_observation(self, profile: IdentityProfile, observation: ProfileObservation) -> None:
        """Commit state and event together; perform no vector arithmetic."""
        if profile.person_id != observation.person_id:
            raise ValueError("Profile and observation must refer to the same person.")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO persons (
                    person_id, mean_embedding, observations, embedding_weight_sum,
                    created_at, last_seen, best_snapshot_path, best_snapshot_quality, embedding_sum,
                    detail_vector, detail_weight_sum
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(person_id) DO UPDATE SET
                    mean_embedding=excluded.mean_embedding, observations=excluded.observations,
                    embedding_weight_sum=excluded.embedding_weight_sum, last_seen=excluded.last_seen,
                    best_snapshot_path=excluded.best_snapshot_path,
                    best_snapshot_quality=excluded.best_snapshot_quality, embedding_sum=excluded.embedding_sum,
                    detail_vector=excluded.detail_vector, detail_weight_sum=excluded.detail_weight_sum
                """,
                (profile.person_id, self._vector_to_json(profile.embedding), profile.observations,
                 profile.embedding_weight_sum, profile.created_at, profile.last_seen,
                 profile.best_snapshot_path, profile.best_snapshot_quality,
                 self._vector_to_json(profile.embedding_sum) if profile.embedding_sum is not None else None,
                 self._vector_to_json(profile.detail_vector) if profile.detail_vector is not None else None,
                 profile.detail_weight_sum),
            )
            conn.execute(
                """
                INSERT INTO events (
                    person_id, source, frame_index, track_id, score, bbox_json,
                    snapshot_path, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (observation.person_id, observation.source, observation.frame_index, observation.track_id,
                 observation.score, json.dumps(list(observation.bbox_xyxy)), observation.snapshot_path,
                 observation.created_at, json.dumps(observation.payload, ensure_ascii=False)),
            )

    def search(self, embedding: np.ndarray, threshold: float,
               exclude_person_ids: set[str] | None = None) -> MatchResult | None:
        """Compatibility facade; use ProfileService in new application code."""
        return ProfileService(self).search(embedding, threshold, exclude_person_ids)

    def add_or_update_person(self, person_id: str, embedding: np.ndarray, source: str,
                             frame_index: int, track_id: int, bbox_xyxy: tuple[int, int, int, int],
                             score: float | None, snapshot_path: str | None,
                             payload: Payload | None = None) -> None:
        """Compatibility facade; no matching/update policy is implemented here."""
        ProfileService(self).add_or_update_person(person_id, embedding, source, frame_index,
                                                  track_id, bbox_xyxy, score, snapshot_path, payload)

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
            details = payload.get("details") if isinstance(payload, dict) else None
            if isinstance(details, dict):
                record["details_label"] = details.get("label")
                record["detail_reliability"] = details.get("reliability")
            explanation = payload.get("match_explanation") if isinstance(payload, dict) else None
            if isinstance(explanation, dict):
                record["match_visual_score"] = explanation.get("visual_score")
                record["match_detail_score"] = explanation.get("detail_score")
                record["match_detail_weight"] = explanation.get("detail_weight")
                record["match_motion_bonus"] = explanation.get("motion_bonus")
                record["decision_zone"] = explanation.get("decision_zone")
                record["match_reason"] = explanation.get("summary")
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
