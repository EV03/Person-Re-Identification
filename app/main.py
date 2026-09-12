"""Command-line entry point for running one person-ReID analysis."""

from __future__ import annotations

import argparse

from app.config import AppPaths
from app.modes.mode_registry import get_mode, list_modes
from app.pipeline.orchestrator import PersonReIdPipeline


def parse_args() -> argparse.Namespace:
    """Build the CLI from the currently available built-in and custom modes."""

    paths = AppPaths()
    available_modes = sorted(list_modes(paths).keys())

    parser = argparse.ArgumentParser(description="Run local person re-identification pipeline on a video file or webcam.")
    parser.add_argument("--source", required=True, help="Video path or webcam index, e.g. 0")
    parser.add_argument("--mode", default="default", choices=available_modes, help="Selectable pipeline mode")
    parser.add_argument("--model", default=None, help="Ultralytics model name/path. Overrides selected mode default.")
    parser.add_argument("--tracker", default=None, help="Tracker config: bytetrack.yaml or botsort.yaml")
    parser.add_argument("--encoder", default=None, choices=["colorhist", "torchreid"], help="ReID encoder backend")
    parser.add_argument("--threshold", type=float, default=None, help="Cosine similarity threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process; 0 processes the full video")
    parser.add_argument("--device", default=None, help="auto, cpu or cuda")
    return parser.parse_args()


def main() -> None:
    """Resolve CLI overrides, run the pipeline and print a compact summary."""

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

    result = pipeline.process(source, progress_callback=progress)
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
