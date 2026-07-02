from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from app.storage.models import MatchResult, PersonRecord, Payload
from app.utils.id_utils import format_person_id, utc_now_iso
from app.utils.detail_utils import (
    combine_visual_and_detail_score,
    compact_detail_influences,
    detail_similarity,
    detail_similarity_breakdown,
)
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

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

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
                    best_snapshot_quality REAL NOT NULL DEFAULT 0.0,
                    detail_vector TEXT,
                    detail_state_json TEXT,
                    detail_weight_sum REAL NOT NULL DEFAULT 0.0
                )
                """
            )
            self._ensure_column(conn, "persons", "embedding_weight_sum", "REAL NOT NULL DEFAULT 1.0")
            self._ensure_column(conn, "persons", "best_snapshot_quality", "REAL NOT NULL DEFAULT 0.0")
            self._ensure_column(conn, "persons", "detail_vector", "TEXT")
            self._ensure_column(conn, "persons", "detail_state_json", "TEXT")
            self._ensure_column(conn, "persons", "detail_weight_sum", "REAL NOT NULL DEFAULT 0.0")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS person_embedding_samples (
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    quality REAL NOT NULL DEFAULT 0.0,
                    source TEXT,
                    frame_index INTEGER,
                    track_id INTEGER,
                    created_at TEXT NOT NULL,
                    payload_json TEXT,
                    FOREIGN KEY(person_id) REFERENCES persons(person_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embedding_samples_person ON person_embedding_samples(person_id)")

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

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS teams (
                    team_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    name TEXT,
                    primary_color TEXT,
                    secondary_color TEXT,
                    metadata_json TEXT,
                    PRIMARY KEY(team_id, run_id),
                    FOREIGN KEY(run_id) REFERENCES analysis_runs(run_id)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS players (
                    player_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    team_id TEXT,
                    display_label TEXT,
                    jersey_number TEXT,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT,
                    PRIMARY KEY(player_id, run_id),
                    FOREIGN KEY(run_id) REFERENCES analysis_runs(run_id)
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS player_frame_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    timestamp_sec REAL,
                    track_id INTEGER,
                    player_id TEXT,
                    team_id TEXT,
                    bbox_json TEXT NOT NULL,
                    pitch_x REAL,
                    pitch_y REAL,
                    speed_mps REAL,
                    distance_delta_m REAL,
                    confidence REAL,
                    payload_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES analysis_runs(run_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_player_frame_events_run ON player_frame_events(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_player_frame_events_player ON player_frame_events(player_id)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ball_frame_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    frame_index INTEGER NOT NULL,
                    timestamp_sec REAL,
                    bbox_json TEXT,
                    pitch_x REAL,
                    pitch_y REAL,
                    speed_mps REAL,
                    nearest_player_id TEXT,
                    nearest_team_id TEXT,
                    confidence REAL,
                    payload_json TEXT,
                    FOREIGN KEY(run_id) REFERENCES analysis_runs(run_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ball_frame_events_run ON ball_frame_events(run_id)")

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS player_stats (
                    run_id TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    team_id TEXT,
                    visible_seconds REAL,
                    distance_m REAL,
                    avg_speed_mps REAL,
                    max_speed_mps REAL,
                    sprint_count INTEGER,
                    ball_near_seconds REAL,
                    possession_seconds REAL,
                    heatmap_json TEXT,
                    payload_json TEXT,
                    PRIMARY KEY(run_id, player_id),
                    FOREIGN KEY(run_id) REFERENCES analysis_runs(run_id)
                )
                """
            )

    @staticmethod
    def _vector_to_json(vector: np.ndarray) -> str:
        return json.dumps(normalize_vector(vector).astype(float).tolist())

    @staticmethod
    def _json_to_vector(value: str) -> np.ndarray:
        return np.asarray(json.loads(value), dtype=np.float32)

    @staticmethod
    def _optional_json_to_vector(value: str | None) -> np.ndarray | None:
        if not value:
            return None
        try:
            return np.asarray(json.loads(value), dtype=np.float32)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    @staticmethod
    def _detail_vector_to_json(vector: np.ndarray | list[float] | None) -> str | None:
        if vector is None:
            return None
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        return json.dumps(arr.astype(float).tolist())

    @staticmethod
    def _detail_vector_from_payload(payload: Payload | None) -> np.ndarray | None:
        if not payload:
            return None
        details = payload.get("details")
        if isinstance(details, dict):
            raw_vector = details.get("vector")
        else:
            raw_vector = payload.get("detail_vector")
        if raw_vector is None:
            return None
        try:
            vector = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None
        return vector if vector.size else None

    @staticmethod
    def _detail_state_from_payload(payload: Payload | None) -> dict[str, object]:
        if not payload:
            return {}
        details = payload.get("details")
        return details if isinstance(details, dict) else {}

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
                    frame_count, width, height, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def search_candidates(
        self,
        embedding: np.ndarray,
        *,
        exclude_person_ids: set[str] | None = None,
        detail_vector: np.ndarray | list[float] | None = None,
        detail_weight: float = 0.0,
        limit: int = 5,
    ) -> list[MatchResult]:
        exclude_person_ids = exclude_person_ids or set()
        query = normalize_vector(embedding)
        query_detail = None if detail_vector is None else np.asarray(detail_vector, dtype=np.float32).reshape(-1)
        best_by_person: dict[str, MatchResult] = {}

        with self._connect() as conn:
            person_rows = conn.execute(
                "SELECT person_id, mean_embedding, detail_vector, best_snapshot_quality FROM persons"
            ).fetchall()
            sample_rows = conn.execute(
                "SELECT person_id, embedding, quality FROM person_embedding_samples"
            ).fetchall()

        person_detail: dict[str, np.ndarray | None] = {}
        for row in person_rows:
            person_id = str(row["person_id"])
            if person_id in exclude_person_ids:
                continue
            stored_detail = self._optional_json_to_vector(row["detail_vector"])
            person_detail[person_id] = stored_detail
            candidate = self._json_to_vector(row["mean_embedding"])
            if candidate.shape != query.shape:
                continue
            visual_score = cosine_similarity(query, candidate)
            detail_breakdown = (
                detail_similarity_breakdown(query_detail, stored_detail)
                if query_detail is not None
                else {"score": 0.5, "reason": "detail_disabled", "features": []}
            )
            detail_score = float(detail_breakdown.get("score", 0.5)) if query_detail is not None else None
            score = combine_visual_and_detail_score(visual_score, detail_score, detail_weight)
            best_by_person[person_id] = MatchResult(
                person_id=person_id,
                score=score,
                is_new=False,
                visual_score=visual_score,
                detail_score=detail_score,
                detail_weight=float(detail_weight),
                detail_breakdown=detail_breakdown,
                reference_type="mean",
                reference_quality=float(row["best_snapshot_quality"] or 0.0),
            )

        for row in sample_rows:
            person_id = str(row["person_id"])
            if person_id in exclude_person_ids:
                continue
            candidate = self._json_to_vector(row["embedding"])
            if candidate.shape != query.shape:
                continue
            visual_score = cosine_similarity(query, candidate)
            stored_detail = person_detail.get(person_id)
            detail_breakdown = (
                detail_similarity_breakdown(query_detail, stored_detail)
                if query_detail is not None
                else {"score": 0.5, "reason": "detail_disabled", "features": []}
            )
            detail_score = float(detail_breakdown.get("score", 0.5)) if query_detail is not None else None
            score = combine_visual_and_detail_score(visual_score, detail_score, detail_weight)
            current = best_by_person.get(person_id)
            if current is None or score > current.score:
                best_by_person[person_id] = MatchResult(
                    person_id=person_id,
                    score=score,
                    is_new=False,
                    visual_score=visual_score,
                    detail_score=detail_score,
                    detail_weight=float(detail_weight),
                    detail_breakdown=detail_breakdown,
                    reference_type="sample",
                    reference_quality=float(row["quality"] or 0.0),
                )

        matches = sorted(best_by_person.values(), key=lambda item: item.score, reverse=True)[: max(int(limit), 1)]
        top_payload = [
            {
                "rank": rank + 1,
                "person_id": match.person_id,
                "score": float(match.score),
                "visual_score": float(match.visual_score) if match.visual_score is not None else None,
                "detail_score": float(match.detail_score) if match.detail_score is not None else None,
                "reference_type": match.reference_type,
                "reference_quality": match.reference_quality,
            }
            for rank, match in enumerate(matches)
        ]
        return [
            MatchResult(
                person_id=match.person_id,
                score=match.score,
                is_new=match.is_new,
                visual_score=match.visual_score,
                detail_score=match.detail_score,
                detail_weight=match.detail_weight,
                detail_breakdown=match.detail_breakdown,
                reference_type=match.reference_type,
                reference_quality=match.reference_quality,
                top_matches=top_payload,
            )
            for match in matches
        ]

    def search(
        self,
        embedding: np.ndarray,
        threshold: float,
        exclude_person_ids: set[str] | None = None,
        detail_vector: np.ndarray | list[float] | None = None,
        detail_weight: float = 0.0,
    ) -> MatchResult | None:
        matches = self.search_candidates(
            embedding,
            exclude_person_ids=exclude_person_ids,
            detail_vector=detail_vector,
            detail_weight=detail_weight,
            limit=5,
        )
        if not matches or matches[0].score < threshold:
            return None
        return matches[0]

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
        detail_vector = self._detail_vector_from_payload(payload)
        detail_state = self._detail_state_from_payload(payload)
        detail_weight = embedding_weight if detail_vector is not None else 0.0
        detail_state_json = json.dumps(detail_state, ensure_ascii=False) if detail_state else None

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT mean_embedding, observations, embedding_weight_sum, best_snapshot_path, best_snapshot_quality,
                       detail_vector, detail_state_json, detail_weight_sum
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
                        created_at, last_seen, best_snapshot_path, best_snapshot_quality,
                        detail_vector, detail_state_json, detail_weight_sum
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        self._detail_vector_to_json(detail_vector),
                        detail_state_json,
                        detail_weight,
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

                old_detail_vector = self._optional_json_to_vector(row["detail_vector"])
                old_detail_weight_sum = float(row["detail_weight_sum"] or 0.0)
                merged_detail_vector = old_detail_vector
                merged_detail_weight_sum = old_detail_weight_sum
                if detail_vector is not None:
                    if old_detail_vector is not None and old_detail_vector.shape == detail_vector.shape and old_detail_weight_sum > 0:
                        merged_detail_weight_sum = old_detail_weight_sum + detail_weight
                        merged_detail_vector = (old_detail_vector * old_detail_weight_sum + detail_vector * detail_weight) / max(merged_detail_weight_sum, 1e-6)
                    else:
                        merged_detail_vector = detail_vector
                        merged_detail_weight_sum = detail_weight

                merged_detail_state_json = detail_state_json or row["detail_state_json"]

                best_snapshot_path = row["best_snapshot_path"] or snapshot_path
                best_snapshot_quality = float(row["best_snapshot_quality"] or 0.0)
                if snapshot_path and snapshot_quality >= best_snapshot_quality:
                    best_snapshot_path = snapshot_path
                    best_snapshot_quality = snapshot_quality

                conn.execute(
                    """
                    UPDATE persons
                    SET mean_embedding = ?, observations = ?, embedding_weight_sum = ?,
                        last_seen = ?, best_snapshot_path = ?, best_snapshot_quality = ?,
                        detail_vector = ?, detail_state_json = ?, detail_weight_sum = ?
                    WHERE person_id = ?
                    """,
                    (
                        self._vector_to_json(mean_embedding),
                        new_observations,
                        new_weight_sum,
                        now,
                        best_snapshot_path,
                        best_snapshot_quality,
                        self._detail_vector_to_json(merged_detail_vector),
                        merged_detail_state_json,
                        merged_detail_weight_sum,
                        person_id,
                    ),
                )

            conn.execute(
                """
                INSERT INTO person_embedding_samples (person_id, embedding, quality, source, frame_index, track_id, created_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    self._vector_to_json(embedding),
                    float(snapshot_quality),
                    source,
                    int(frame_index),
                    int(track_id),
                    now,
                    payload_json,
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
            record["decision_zone"] = payload.get("decision_zone")
            record["created_new_person"] = payload.get("created_new_person")
            record["calibration_mode"] = payload.get("calibration_mode")
            record["calibration_label"] = payload.get("calibration_label")
            record["quality_score"] = payload.get("quality_score")
            record["quality_average"] = payload.get("quality_average")
            record["good_frame_count"] = payload.get("good_frame_count")
            record["motion_direction"] = payload.get("motion_direction")
            record["motion_speed_px_per_sec"] = payload.get("motion_speed_px_per_sec")
            record["motion_plausibility_score"] = payload.get("motion_plausibility_score")
            record["motion_is_large_jump"] = payload.get("motion_is_large_jump")
            details = payload.get("details") if isinstance(payload, dict) else None
            if isinstance(details, dict):
                record["details_label"] = details.get("label")
                record["detail_reliability"] = details.get("reliability")
                features = details.get("features")
                if isinstance(features, dict):
                    visible = []
                    for name, feature in features.items():
                        if isinstance(feature, dict) and feature.get("status") not in {None, "unknown", "observed"}:
                            visible.append(f"{name}:{feature.get('status')}")
                    record["details_binary"] = ", ".join(visible[:8])
            match_explanation = payload.get("match_explanation") if isinstance(payload, dict) else None
            if isinstance(match_explanation, dict):
                record["match_visual_score"] = match_explanation.get("visual_score")
                record["match_detail_score"] = match_explanation.get("detail_score")
                record["match_detail_weight"] = match_explanation.get("detail_weight")
                record["match_reason"] = match_explanation.get("summary")
                record["top_matches"] = json.dumps(match_explanation.get("top_matches", []), ensure_ascii=False)
                record["reference_type"] = match_explanation.get("reference_type")
            else:
                top_matches = payload.get("top_matches") if isinstance(payload, dict) else None
                if top_matches is not None:
                    record["top_matches"] = json.dumps(top_matches, ensure_ascii=False)
            records.append(record)

        return pd.DataFrame(records)

    def analysis_runs_dataframe(self, limit: int = 100) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, mode_id, mode_name, pipeline_type, source, fps, frame_count, width, height, created_at
                FROM analysis_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])
