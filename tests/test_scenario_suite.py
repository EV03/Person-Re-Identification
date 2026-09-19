from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.evaluation.metrics import compute_scenario_diagnostics
from app.config import AppPaths
from app.evaluation.scenario_suite import (
    ScenarioVideo,
    aggregate_rows,
    discover_scenario_videos,
    load_scenario_manifest,
    run_scenario_suite,
    write_scenario_manifest,
    write_suite_reports,
)


class ScenarioSuiteTests(unittest.TestCase):
    def test_exactly_one_video_per_g_folder_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            files = [
                root / "G1" / "no_crossing.mp4",
                root / "g2" / "leave_and_return.mp4",
                root / "G3" / "similar_clothes.mp4",
                root / "G4" / "crossing.mp4",
                root / "Default" / "ignored.mp4",
            ]
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            entries = discover_scenario_videos(root)
            self.assertEqual(len(entries), 4)
            self.assertEqual([entry.sequence for entry in entries], ["g1", "g2", "g3", "g4"])
            manifest = write_scenario_manifest(entries, root / "manifest.csv")
            restored = load_scenario_manifest(manifest)
            self.assertEqual([(item.group, item.sequence) for item in restored],
                             [(item.group, item.sequence) for item in entries])

    def test_duplicate_or_missing_group_video_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative in ("G1/one.mp4", "G1/two.mp4", "G2/one.mp4", "G3/one.mp4", "G4/one.mp4"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"video")
            with self.assertRaisesRegex(ValueError, "Exactly one video"):
                discover_scenario_videos(root)

    def test_multi_person_diagnostics_are_explicitly_not_ground_truth_metrics(self) -> None:
        events = [
            {"frame_index": 1, "track_id": 1, "person_id": "p1", "state": "matched_identity", "decision_zone": "strong"},
            {"frame_index": 2, "track_id": 1, "person_id": "p2", "state": "pending_weak_match", "decision_zone": "weak"},
            {"frame_index": 2, "track_id": 2, "person_id": "p2", "state": "known_track", "decision_zone": None},
        ]
        metrics = compute_scenario_diagnostics(events, processed_frames=2, expected_real_person_count=2)
        self.assertEqual(metrics["track_to_person_output_switches"], 1)
        self.assertEqual(metrics["person_to_track_fragment_surplus"], 1)
        self.assertEqual(metrics["profile_count_delta"], 0)
        self.assertFalse(metrics["ground_truth_identity_metrics_available"])

    def test_master_and_separate_profile_reports_are_written(self) -> None:
        rows = [{
            "group": "G1", "group_description": "test", "sequence": "s1", "video_name": "one.mp4",
            "source": "one.mp4", "expected_person_count": 2, "notes": "", "mode_id": "default",
            "mode_name": "B0", "repetition": 1, "status": "completed", "experiment_manifest": "e.json",
            "run_manifest": "m.json", "predictions": "frames.jsonl", "error": None,
            "main_metrics": {"processed_frames": 10, "frames_per_second": 5.0, "real_time_factor": 2.0},
            "identity_diagnostics": {"person_assignment_ratio": .8, "profile_count_delta": 0,
                                     "track_to_person_output_switches": 0,
                                     "person_to_track_fragment_surplus": 1,
                                     "unique_track_ids": 3, "unique_person_ids": 2,
                                     "strong_match_count": 1, "weak_match_count": 2, "low_match_count": 3},
        }]
        with tempfile.TemporaryDirectory() as folder:
            reports = write_suite_reports(rows, Path(folder))
            self.assertTrue(all(path.exists() for path in reports.values()))
            self.assertTrue((reports["profiles"] / "default.md").is_file())
            self.assertTrue((reports["profiles"] / "default.json").is_file())
            self.assertTrue((reports["profiles"] / "default.csv").is_file())
            self.assertEqual(json.loads(reports["json"].read_text(encoding="utf-8"))["rows"][0]["group"], "G1")
            self.assertEqual(aggregate_rows(rows)[0]["mean_fps"], 5.0)

    def test_suite_runner_combines_main_and_identity_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "G1" / "clip.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            counter = 0

            def fake_run_unit(sources, config, *, root, base_paths):
                nonlocal counter
                counter += 1
                unit = Path(root) / f"unit_{counter}"
                unit.mkdir(parents=True)
                runs = []
                for index, _source in enumerate(sources):
                    run_dir = unit / f"run_{index}"
                    run_dir.mkdir()
                    predictions = run_dir / "frames.jsonl"
                    predictions.write_text(json.dumps({
                        "frame_index": 1,
                        "detections": [{"track_id": 1, "person_id": "p1",
                                        "state": "matched_identity", "decision_zone": "strong"}],
                    }) + "\n", encoding="utf-8")
                    manifest = run_dir / "manifest.json"
                    manifest.write_text(json.dumps({
                        "status": "completed", "processed_frames": 1,
                        "timings": {"processing_seconds": .5, "frames_per_second": 2.0,
                                    "real_time_factor": .5},
                        "exports": {"frames_jsonl": str(predictions), "tracking_mot": "tracking.txt"},
                    }), encoding="utf-8")
                    runs.append({"status": "completed", "manifest": str(manifest),
                                 "predictions": str(predictions)})
                experiment = unit / "experiment.json"
                experiment.write_text(json.dumps({"runs": runs}), encoding="utf-8")
                return experiment

            paths = AppPaths(db_path=root / "db.sqlite3", snapshot_dir=root / "snapshots",
                             output_dir=root / "output", input_dir=root / "input",
                             mode_dir=root / "modes")
            reports = run_scenario_suite(
                [ScenarioVideo("G1", "clip", video, 2, "test")],
                mode_ids=["default"], output_root=root / "suite", base_paths=paths,
                run_unit_fn=fake_run_unit,
            )
            payload = json.loads(reports["json"].read_text(encoding="utf-8"))
            row = payload["rows"][0]
            self.assertEqual(row["main_metrics"]["frames_per_second"], 2.0)
            self.assertEqual(row["identity_diagnostics"]["strong_match_count"], 1)


if __name__ == "__main__":
    unittest.main()
