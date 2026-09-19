"""Create and run a G1-G4 multi-preset evaluation manifest.

Examples:
    python scripts/evaluate_g1_g4_suite.py init-manifest --video-root Test-daten
    python scripts/evaluate_g1_g4_suite.py run --manifest data/g1_g4_manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppPaths  # noqa: E402
from app.evaluation.scenario_suite import (  # noqa: E402
    DEFAULT_SUITE_MODES,
    discover_scenario_videos,
    load_scenario_manifest,
    resolve_project_path,
    run_scenario_suite,
    write_scenario_manifest,
)
from app.modes.mode_registry import list_modes  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G1-G4 evaluation across Main and Details presets.")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-manifest", help="Scan G1/G2/G3/G4 video folders.")
    init.add_argument("--video-root", default="Test-daten")
    init.add_argument("--output", default="data/g1_g4_manifest.csv")
    run = sub.add_parser("run", help="Run all requested presets and write comparison reports.")
    run.add_argument("--manifest", default="data/g1_g4_manifest.csv")
    run.add_argument("--output-dir", default="data/evaluation_suites")
    run.add_argument("--modes", nargs="+", choices=sorted(list_modes(AppPaths())),
                     default=list(DEFAULT_SUITE_MODES))
    run.add_argument("--repetitions", type=int, default=1)
    run.add_argument("--checkpoint", default=None)
    run.add_argument("--device", default=None)
    run.add_argument("--max-frames", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "init-manifest":
        entries = discover_scenario_videos(resolve_project_path(args.video_root))
        output = write_scenario_manifest(entries, resolve_project_path(args.output))
        print(f"{output} ({len(entries)} videos)")
        return
    reports = run_scenario_suite(
        load_scenario_manifest(resolve_project_path(args.manifest)),
        mode_ids=args.modes,
        output_root=resolve_project_path(args.output_dir),
        repetitions=args.repetitions,
        checkpoint=args.checkpoint,
        device=args.device,
        max_frames=args.max_frames,
    )
    for name, path in reports.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
