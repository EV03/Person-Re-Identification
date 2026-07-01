from __future__ import annotations

import os
import sys
from pathlib import Path

from app.config import PROJECT_ROOT


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main() -> int:
    print(f"Python executable: {sys.executable}")
    try:
        import qdrant_client
        from qdrant_client import QdrantClient
    except Exception as exc:
        print(f"qdrant-client import failed: {type(exc).__name__}: {exc}")
        print("Install into the active venv with: python -m pip install qdrant-client")
        return 1

    version = getattr(qdrant_client, "__version__", "unknown")
    print(f"qdrant-client import ok: {version}")

    mode = os.getenv("QDRANT_MODE", "local").strip().lower()
    print(f"Qdrant mode: {mode}")

    try:
        if mode in {"local", "embedded", "file"}:
            local_path = _resolve_project_path(os.getenv("QDRANT_LOCAL_PATH", "data/qdrant_local"))
            local_path.mkdir(parents=True, exist_ok=True)
            print(f"Qdrant local path: {local_path}")
            client = QdrantClient(path=str(local_path))
        elif mode in {"memory", "in_memory", ":memory:"}:
            print("Qdrant in-memory mode: no data is persisted after process shutdown.")
            client = QdrantClient(":memory:")
        else:
            url = os.getenv("QDRANT_URL", "http://localhost:6333")
            prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "false").lower() in {"1", "true", "yes", "on"}
            print(f"Qdrant URL: {url}")
            print(f"Prefer gRPC: {prefer_grpc}")
            client = QdrantClient(url=url, prefer_grpc=prefer_grpc)

        collections = client.get_collections()
        print("Qdrant check ok.")
        print(collections)
        close = getattr(client, "close", None)
        if callable(close):
            close()
        return 0
    except Exception as exc:
        print(f"Qdrant check failed: {type(exc).__name__}: {exc}")
        if mode not in {"local", "embedded", "file", "memory", "in_memory", ":memory:"}:
            print("No Qdrant server is reachable. Start one, or set QDRANT_MODE=local to run without Docker.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
