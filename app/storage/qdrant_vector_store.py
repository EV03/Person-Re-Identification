from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np

from app.storage.models import MatchResult, Payload
from app.storage.vector_store import SQLiteVectorStore
from app.utils.id_utils import format_person_id, utc_now_iso
from app.utils.detail_utils import combine_visual_and_detail_score, detail_similarity_breakdown
from app.utils.image_utils import normalize_vector


class QdrantVectorStore:
    """Qdrant-backed vector store with SQLite metadata compatibility.

    Qdrant is used for the actual person embedding collection and nearest-neighbour
    search. The existing SQLite store is kept as a lightweight metadata/event log so
    the current Streamlit tables and analysis-run history continue to work.
    """

    def __init__(
        self,
        metadata_db_path: Path,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        collection_name: str = "person_reid_embeddings",
        prefer_grpc: bool = False,
        mode: str = "server",
        local_path: Path | str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient, models
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            missing_module = exc.name or "unknown"
            if missing_module == "qdrant_client" or missing_module.startswith("qdrant_client."):
                raise RuntimeError(
                    "Qdrant backend requested, but qdrant-client is not installed in the "
                    "Python environment that runs Streamlit. Install it with: "
                    "python -m pip install qdrant-client"
                ) from exc
            raise RuntimeError(
                "qdrant-client is installed, but one of its dependencies could not be imported: "
                f"{missing_module}. Original error: {type(exc).__name__}: {exc}"
            ) from exc
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "qdrant-client is installed, but importing it failed. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        self.metadata_store = SQLiteVectorStore(metadata_db_path)
        self.collection_name = collection_name
        self.mode = (mode or "server").strip().lower()
        self.url = url
        self.prefer_grpc = bool(prefer_grpc)
        self.local_path = Path(local_path) if local_path else None

        if self.mode in {"local", "embedded", "file"}:
            if self.local_path is None:
                raise RuntimeError("Qdrant local mode needs a local storage path.")
            self.local_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(self.local_path))
        elif self.mode in {"memory", "in_memory", ":memory:"}:
            self.client = QdrantClient(":memory:")
        elif self.mode in {"server", "remote", "http"}:
            self.client = QdrantClient(url=url, api_key=api_key or None, prefer_grpc=self.prefer_grpc)
        else:
            raise RuntimeError(
                f"Unknown Qdrant mode '{mode}'. Use 'local', 'memory' or 'server'."
            )

        self.models = models

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
        metadata_close = getattr(self.metadata_store, "close", None)
        if callable(metadata_close):
            metadata_close()

    @staticmethod
    def _point_id(collection_name: str, person_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"person-reid:{collection_name}:{person_id}"))

    @staticmethod
    def _point_vector(point: Any) -> np.ndarray | None:
        vector = getattr(point, "vector", None)
        if isinstance(vector, dict):
            if not vector:
                return None
            vector = next(iter(vector.values()))
        if vector is None:
            return None
        return np.asarray(vector, dtype=np.float32)

    @staticmethod
    def _score(point: Any) -> float:
        raw_score = getattr(point, "score", 0.0)
        try:
            return float(raw_score)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _payload(point: Any) -> dict[str, Any]:
        payload = getattr(point, "payload", None)
        return payload if isinstance(payload, dict) else {}

    def _connection_error(self, exc: Exception) -> RuntimeError:
        if self.mode in {"local", "embedded", "file"}:
            return RuntimeError(
                f"Qdrant local store at '{self.local_path}' could not be opened or queried. "
                "Close other running Streamlit sessions that use the same local Qdrant path, "
                "or choose another local path. "
                f"Original error: {type(exc).__name__}: {exc}"
            )
        if self.mode in {"memory", "in_memory", ":memory:"}:
            return RuntimeError(
                "Qdrant in-memory store could not be queried. "
                f"Original error: {type(exc).__name__}: {exc}"
            )

        transport = "gRPC" if self.prefer_grpc else "HTTP"
        return RuntimeError(
            f"Qdrant server is not reachable at '{self.url}' via {transport}. "
            "No Qdrant server is running. Either start Qdrant with Docker/binary, "
            "or switch Qdrant mode to 'local' to run without Docker. "
            "Then test it with: python scripts/check_qdrant_setup.py. "
            f"Original error: {type(exc).__name__}: {exc}"
        )

    def _collection_exists(self) -> bool:
        try:
            return bool(self.client.collection_exists(self.collection_name))
        except AttributeError:
            try:
                self.client.get_collection(self.collection_name)
                return True
            except Exception as exc:
                message = str(exc).lower()
                if "not found" in message or "doesn't exist" in message or "does not exist" in message:
                    return False
                raise self._connection_error(exc) from exc
        except Exception as exc:
            raise self._connection_error(exc) from exc

    def _ensure_collection(self, vector_size: int) -> None:
        if not self._collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.models.VectorParams(
                    size=int(vector_size),
                    distance=self.models.Distance.COSINE,
                ),
            )
            return

        try:
            collection = self.client.get_collection(self.collection_name)
            vectors_config = collection.config.params.vectors
            configured_size = getattr(vectors_config, "size", None)
            if configured_size is not None and int(configured_size) != int(vector_size):
                raise ValueError(
                    f"Qdrant collection '{self.collection_name}' expects {configured_size}D vectors, "
                    f"but the current encoder produced {vector_size}D vectors. "
                    "Use a separate collection or delete/recreate the existing one."
                )
        except AttributeError:
            return

    def _retrieve_person_point(self, person_id: str) -> Any | None:
        if not self._collection_exists():
            return None
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[self._point_id(self.collection_name, person_id)],
            with_vectors=True,
            with_payload=True,
        )
        return points[0] if points else None

    def _query_points(self, query: np.ndarray, limit: int) -> list[Any]:
        if not self._collection_exists():
            return []

        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query.astype(float).tolist(),
                limit=limit,
                with_payload=True,
            )
            return list(getattr(response, "points", response))
        except AttributeError:
            return list(
                self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query.astype(float).tolist(),
                    limit=limit,
                    with_payload=True,
                )
            )

    def count_persons(self) -> int:
        if not self._collection_exists():
            return 0
        response = self.client.count(collection_name=self.collection_name, exact=True)
        return int(getattr(response, "count", 0))

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
        metadata = dict(metadata or {})
        metadata["vector_store_backend"] = "qdrant"
        metadata["qdrant_collection"] = self.collection_name
        metadata["qdrant_mode"] = self.mode
        metadata["qdrant_local_path"] = str(self.local_path) if self.local_path else None
        self.metadata_store.add_analysis_run(
            run_id=run_id,
            mode_id=mode_id,
            mode_name=mode_name,
            pipeline_type=pipeline_type,
            source=source,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            metadata=metadata,
        )

    def search(
        self,
        embedding: np.ndarray,
        threshold: float,
        exclude_person_ids: set[str] | None = None,
        detail_vector: np.ndarray | list[float] | None = None,
        detail_weight: float = 0.0,
    ) -> MatchResult | None:
        exclude_person_ids = exclude_person_ids or set()
        query = normalize_vector(embedding)
        query_detail = None if detail_vector is None else np.asarray(detail_vector, dtype=np.float32).reshape(-1)
        points = self._query_points(query, limit=max(len(exclude_person_ids) + 5, 10))

        best_person_id: str | None = None
        best_score = -1.0
        best_visual_score: float | None = None
        best_detail_score: float | None = None
        best_detail_breakdown: dict[str, object] = {}
        for point in points:
            payload = self._payload(point)
            person_id = str(payload.get("person_id", ""))
            if not person_id or person_id in exclude_person_ids:
                continue
            visual_score = self._score(point)
            stored_detail = payload.get("detail_vector")
            detail_breakdown = (
                detail_similarity_breakdown(query_detail, stored_detail)
                if query_detail is not None
                else {"score": 0.5, "reason": "detail_disabled", "features": []}
            )
            detail_score = float(detail_breakdown.get("score", 0.5)) if query_detail is not None else None
            score = combine_visual_and_detail_score(visual_score, detail_score, detail_weight)
            if score > best_score:
                best_person_id = person_id
                best_score = score
                best_visual_score = visual_score
                best_detail_score = detail_score
                best_detail_breakdown = detail_breakdown

        if best_person_id is not None and best_score >= threshold:
            return MatchResult(
                person_id=best_person_id,
                score=best_score,
                is_new=False,
                visual_score=best_visual_score,
                detail_score=best_detail_score,
                detail_weight=float(detail_weight),
                detail_breakdown=best_detail_breakdown,
            )
        return None

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
        payload = dict(payload or {})
        embedding = normalize_vector(embedding)
        embedding_weight = SQLiteVectorStore._embedding_weight_from_payload(payload)
        snapshot_quality = SQLiteVectorStore._snapshot_quality_from_payload(payload)
        detail_vector = SQLiteVectorStore._detail_vector_from_payload(payload)
        detail_weight = embedding_weight if detail_vector is not None else 0.0
        detail_state = SQLiteVectorStore._detail_state_from_payload(payload)
        point_id = self._point_id(self.collection_name, person_id)
        existing_point = self._retrieve_person_point(person_id)

        if existing_point is None:
            mean_embedding = embedding
            observations = 1
            weight_sum = embedding_weight
            created_at = now
            best_snapshot_path = snapshot_path
            best_snapshot_quality = snapshot_quality
        else:
            existing_payload = self._payload(existing_point)
            old_embedding = self._point_vector(existing_point)
            old_observations = int(existing_payload.get("observations", 1) or 1)
            old_weight_sum = float(existing_payload.get("embedding_weight_sum", max(old_observations, 1)) or 1.0)
            observations = old_observations + 1
            weight_sum = old_weight_sum + embedding_weight
            created_at = str(existing_payload.get("created_at") or now)

            if old_embedding is not None and old_embedding.shape == embedding.shape:
                mean_embedding = normalize_vector(
                    (old_embedding * old_weight_sum + embedding * embedding_weight) / weight_sum
                )
            else:
                mean_embedding = embedding
                weight_sum = embedding_weight

            best_snapshot_path = existing_payload.get("best_snapshot_path") or snapshot_path
            best_snapshot_quality = float(existing_payload.get("best_snapshot_quality", 0.0) or 0.0)
            if snapshot_path and snapshot_quality >= best_snapshot_quality:
                best_snapshot_path = snapshot_path
                best_snapshot_quality = snapshot_quality

        existing_detail_vector = None
        existing_detail_weight_sum = 0.0
        if existing_point is not None:
            existing_payload = self._payload(existing_point)
            raw_existing_detail = existing_payload.get("detail_vector")
            if raw_existing_detail is not None:
                try:
                    existing_detail_vector = np.asarray(raw_existing_detail, dtype=np.float32).reshape(-1)
                except (TypeError, ValueError):
                    existing_detail_vector = None
            existing_detail_weight_sum = float(existing_payload.get("detail_weight_sum", 0.0) or 0.0)

        merged_detail_vector = existing_detail_vector
        merged_detail_weight_sum = existing_detail_weight_sum
        if detail_vector is not None:
            if existing_detail_vector is not None and existing_detail_vector.shape == detail_vector.shape and existing_detail_weight_sum > 0:
                merged_detail_weight_sum = existing_detail_weight_sum + detail_weight
                merged_detail_vector = (existing_detail_vector * existing_detail_weight_sum + detail_vector * detail_weight) / max(merged_detail_weight_sum, 1e-6)
            else:
                merged_detail_vector = detail_vector
                merged_detail_weight_sum = detail_weight

        self._ensure_collection(vector_size=int(mean_embedding.shape[0]))
        qdrant_payload = {
            "person_id": person_id,
            "observations": int(observations),
            "embedding_weight_sum": float(weight_sum),
            "created_at": created_at,
            "last_seen": now,
            "best_snapshot_path": best_snapshot_path,
            "best_snapshot_quality": float(best_snapshot_quality),
            "last_source": source,
            "last_frame_index": int(frame_index),
            "last_track_id": int(track_id),
            "last_score": score,
            "last_bbox_xyxy": list(bbox_xyxy),
            "last_event_payload": payload,
            "detail_vector": merged_detail_vector.astype(float).tolist() if merged_detail_vector is not None else None,
            "detail_state": detail_state or None,
            "detail_weight_sum": float(merged_detail_weight_sum),
        }
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                self.models.PointStruct(
                    id=point_id,
                    vector=mean_embedding.astype(float).tolist(),
                    payload=qdrant_payload,
                )
            ],
        )

        self.metadata_store.add_or_update_person(
            person_id=person_id,
            embedding=mean_embedding,
            source=source,
            frame_index=frame_index,
            track_id=track_id,
            bbox_xyxy=bbox_xyxy,
            score=score,
            snapshot_path=snapshot_path,
            payload=payload,
        )

    def list_persons(self):
        return self.metadata_store.list_persons()

    def persons_dataframe(self):
        return self.metadata_store.persons_dataframe()

    def events_dataframe(self, limit: int = 200):
        return self.metadata_store.events_dataframe(limit=limit)

    def analysis_runs_dataframe(self, limit: int = 100):
        return self.metadata_store.analysis_runs_dataframe(limit=limit)
