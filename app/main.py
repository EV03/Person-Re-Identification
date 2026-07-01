from __future__ import annotations

import argparse
from pathlib import Path

from app.config import AppPaths
from app.modes.mode_registry import get_mode, list_modes
from app.pipeline.orchestrator import PersonReIdPipeline


def parse_args() -> argparse.Namespace:
    paths = AppPaths()
    available_modes = sorted(list_modes(paths).keys())

    parser = argparse.ArgumentParser(description="Run local person re-identification pipeline on a video file or webcam.")
    parser.add_argument("--source", required=True, help="Video path or webcam index, e.g. 0")
    parser.add_argument("--mode", default="default", choices=available_modes, help="Selectable pipeline mode")
    parser.add_argument("--model", default=None, help="Ultralytics model name/path. Overrides selected mode default.")
    parser.add_argument("--tracker", default=None, help="Tracker config: bytetrack.yaml or botsort.yaml")
    parser.add_argument("--encoder", default=None, choices=["colorhist", "torchreid"], help="ReID encoder backend")
    parser.add_argument("--store", default=None, choices=["sqlite", "qdrant"], help="Vector store backend")
    parser.add_argument("--qdrant-url", default=None, help="Qdrant URL, e.g. http://localhost:6333")
    parser.add_argument("--qdrant-collection", default=None, help="Qdrant collection name")
    parser.add_argument("--qdrant-mode", default=None, choices=["local", "server", "memory"], help="Qdrant mode. Use local to run without Docker/server.")
    parser.add_argument("--qdrant-local-path", default=None, help="Local Qdrant storage path when --qdrant-mode local is used")
    parser.add_argument("--qdrant-grpc", action="store_true", help="Prefer Qdrant gRPC transport")
    parser.add_argument(
        "--disable-internal-motion-with-botsort",
        action="store_true",
        help="Disable the internal direction tracker when BoT-SORT is selected.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Cosine similarity threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--device", default=None, help="auto, cpu or cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source: str | int
    source = int(args.source) if args.source.isdigit() else args.source

    paths = AppPaths()
    mode = get_mode(args.mode, paths)
    overrides = {}
    if args.model is not None:
        overrides["yolo_model"] = args.model
    if args.tracker is not None:
        overrides["tracker"] = args.tracker
    if args.encoder is not None:
        overrides["encoder_backend"] = args.encoder
    if args.store is not None:
        overrides["vector_store_backend"] = args.store
    if args.qdrant_url is not None:
        overrides["qdrant_url"] = args.qdrant_url
    if args.qdrant_collection is not None:
        overrides["qdrant_collection"] = args.qdrant_collection
    if args.qdrant_mode is not None:
        overrides["qdrant_mode"] = args.qdrant_mode
    if args.qdrant_local_path is not None:
        overrides["qdrant_local_path"] = args.qdrant_local_path
    if args.qdrant_grpc:
        overrides["qdrant_prefer_grpc"] = True
    if args.disable_internal_motion_with_botsort:
        overrides["disable_internal_motion_when_botsort"] = True
        if (args.tracker or mode.tracker).lower().endswith("botsort.yaml"):
            overrides["enable_motion_analysis"] = False
            overrides["draw_motion_vectors"] = False
    if args.threshold is not None:
        overrides["match_threshold"] = args.threshold
    if args.max_frames is not None:
        overrides["max_frames"] = args.max_frames
    if args.device is not None:
        overrides["device"] = args.device

    config = mode.to_pipeline_config(**overrides)
    pipeline = PersonReIdPipeline(config=config, paths=paths)

    def progress(current: int, total: int | None, message: str) -> None:
        suffix = f"/{total}" if total else ""
        print(f"[{current}{suffix}] {message}")

    try:
        result = pipeline.process(source, progress_callback=progress)
    finally:
        pipeline.close()
    print("\nDone")
    print(f"Mode: {result.mode_name} ({result.mode_id})")
    print(f"Run ID: {result.run_id}")
    print(f"Output video: {result.output_video_path}")
    print(f"Processed frames: {result.processed_frames}")
    print(f"Created persons: {result.created_persons}")
    print(f"Matched events: {result.matched_events}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
