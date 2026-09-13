"""Example: python -m app.evaluation --sources video.mp4 --modes default colorhist --repetitions 3"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import AppPaths
from app.evaluation.runner import run_unit
from app.modes.mode_registry import get_mode, list_modes


def main() -> None:
    paths = AppPaths()
    parser = argparse.ArgumentParser(description="Isolated ReID runs; all supplied sources belong to ONE scenario/unit.")
    parser.add_argument("--sources", nargs="+", required=True, type=Path,
                        help="One clip, or related registration/return clips sharing profiles within one test sequence.")
    parser.add_argument("--modes", nargs="+", choices=sorted(list_modes(paths)),
                        default=["default", "colorhist", "no_quality_thresholds", "no_update_similarity"])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    failed = False
    for mode_id in args.modes:
        overrides = {"max_frames": 0}
        if args.checkpoint is not None:
            overrides["reid_checkpoint"] = args.checkpoint
        if args.device is not None:
            overrides["device"] = args.device
        config = get_mode(mode_id, paths).to_pipeline_config(**overrides)
        for repetition in range(1, args.repetitions + 1):
            try:
                manifest = run_unit(args.sources, config, root=args.root, base_paths=paths)
                print(f"{mode_id} #{repetition}: {manifest}")
            except Exception as exc:
                failed = True
                print(f"FAILED {mode_id} #{repetition}: {exc}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
