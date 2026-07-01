from __future__ import annotations

from pathlib import Path

from app.config import AppPaths, PipelineConfig, PROJECT_ROOT
from app.storage.qdrant_vector_store import QdrantVectorStore
from app.storage.vector_store import SQLiteVectorStore


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_vector_store(config: PipelineConfig, paths: AppPaths):
    """Build the configured vector store backend."""
    backend = (config.vector_store_backend or "sqlite").strip().lower()
    if backend == "sqlite":
        return SQLiteVectorStore(paths.db_path)
    if backend == "qdrant":
        return QdrantVectorStore(
            metadata_db_path=paths.db_path,
            url=config.qdrant_url,
            api_key=config.qdrant_api_key or None,
            collection_name=config.qdrant_collection,
            prefer_grpc=bool(config.qdrant_prefer_grpc),
            mode=config.qdrant_mode,
            local_path=_resolve_project_path(config.qdrant_local_path),
        )
    raise ValueError(f"Unknown vector store backend: {config.vector_store_backend}")
