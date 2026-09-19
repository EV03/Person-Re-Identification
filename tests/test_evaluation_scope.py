from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from dataclasses import asdict
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.config import AppPaths, PipelineConfig
from app.modes.base_mode import ModeConfig
from app.modes.mode_registry import builtin_modes, list_modes, save_custom_mode
from app.pipeline.orchestrator import PersonReIdPipeline
from app.storage.models import Detection, MatchResult
from app.reid.repository import ProfileUpdateDecision
from app.storage.vector_store import SQLiteVectorStore
from tests.test_pipeline_resources import FakeCapture, FakeWriter


def paths_for(base: Path) -> AppPaths:
    return AppPaths(
        db_path=base / "db.sqlite3",
        snapshot_dir=base / "snapshots",
        output_dir=base / "output",
        input_dir=base / "input",
        mode_dir=base / "modes",
    )


class EvaluationPresetTests(unittest.TestCase):
    def test_a1_a2_and_a3_change_only_the_documented_parameters(self) -> None:
        modes = builtin_modes()
        base = asdict(modes["default"].to_pipeline_config())
        for name, expected in (
            ("colorhist", {"encoder_backend"}),
            ("no_quality_thresholds", {"min_embedding_quality", "min_update_quality"}),
            ("no_update_similarity", {"min_update_similarity"}),
        ):
            with self.subTest(mode=name):
                variant = asdict(modes[name].to_pipeline_config())
                changed = {key for key in base if base[key] != variant[key]}
                self.assertEqual(changed - {"mode_id", "mode_name"}, expected)
        self.assertEqual(modes["colorhist"].encoder_backend, "colorhist")
        self.assertEqual(modes["no_quality_thresholds"].min_embedding_quality, 0)
        self.assertEqual(modes["no_quality_thresholds"].min_update_quality, 0)
        self.assertEqual(modes["no_update_similarity"].min_update_similarity, -1)

    def test_custom_preset_round_trip_preserves_name_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            mode = ModeConfig(mode_id="pilot", name="Pilot 1", description="test")
            save_custom_mode(mode, paths)
            restored = list_modes(paths)["pilot"].to_pipeline_config(max_frames=0)
            self.assertEqual(restored.mode_name, "Pilot 1")
            self.assertEqual(restored.max_frames, 0)

    def test_old_presets_are_preserved_but_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            paths.ensure()
            legacy = paths.mode_dir / "custom_modes.json"
            contents = json.dumps({"modes": [{"mode_id": "football", "pipeline_type": "football_analysis"}]})
            legacy.write_text(contents, encoding="utf-8")
            self.assertEqual(set(list_modes(paths)), set(builtin_modes()))
            save_custom_mode(ModeConfig("pilot", "Pilot", "test"), paths)
            self.assertEqual(legacy.read_text(encoding="utf-8"), contents)

    def test_other_pipeline_types_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ModeConfig("legacy", "Legacy", "test", pipeline_type="football_analysis")
        with self.assertRaises(ValueError):
            PipelineConfig(pipeline_type="football_analysis")

    def test_saved_preset_cannot_shadow_the_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            paths.ensure()
            rogue = ModeConfig("default", "Replaced", "test", encoder_backend="colorhist")
            paths.mode_config_path.write_text(json.dumps({"modes": [asdict(rogue)]}), encoding="utf-8")
            self.assertEqual(list_modes(paths)["default"].encoder_backend, "torchreid")


class ReIdStoreScopeTests(unittest.TestCase):
    def test_new_store_creates_only_reid_tables(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reid.sqlite3"
            SQLiteVectorStore(path)
            with closing(sqlite3.connect(path)) as conn, conn:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertEqual(tables - {"sqlite_sequence"}, {"persons", "events", "analysis_runs"})

    def test_opening_legacy_store_does_not_delete_extension_data(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.sqlite3"
            with closing(sqlite3.connect(path)) as conn, conn:
                conn.execute("CREATE TABLE teams (name TEXT)")
                conn.execute("INSERT INTO teams VALUES ('preserved')")
            store = SQLiteVectorStore(path)
            self.assertEqual(store.count_persons(), 0)
            with closing(sqlite3.connect(path)) as conn, conn:
                self.assertEqual(conn.execute("SELECT name FROM teams").fetchone()[0], "preserved")

    def test_failed_transaction_rolls_back_and_closes_connection(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = SQLiteVectorStore(Path(folder) / "reid.sqlite3")
            with self.assertRaisesRegex(RuntimeError, "rollback"):
                with store._connect() as connection:
                    connection.execute("INSERT INTO persons (person_id, mean_embedding, created_at, last_seen) VALUES ('test', '[]', '', '')")
                    raise RuntimeError("rollback")
            self.assertEqual(store.count_persons(), 0)
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")


class RecordingEncoder:
    def __init__(self) -> None:
        self.crops: list[np.ndarray] = []

    def encode(self, crop: np.ndarray) -> np.ndarray:
        self.crops.append(crop.copy())
        return np.asarray([1.0, 0.0], dtype=np.float32)


class SequenceTracker:
    def __init__(self, frames: list[list[Detection]]) -> None:
        self.frames = iter(frames)

    def track_frame(self, _frame: np.ndarray) -> list[Detection]:
        return next(self.frames)


class MemoryStore:
    def finish_analysis_run(self, *_args, **_kwargs) -> None:
        return None

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.observations: list[dict] = []
        self.metadata: dict = {}

    def add_analysis_run(self, **kwargs) -> None:
        self.metadata = kwargs["metadata"]

    def search(self, _embedding, threshold, exclude_person_ids):
        for person_id in self.ids:
            if person_id not in exclude_person_ids:
                return MatchResult(person_id, 1.0, False)
        return None

    def create_person_id(self) -> str:
        person_id = f"person_{len(self.ids) + 1:06d}"
        self.ids.append(person_id)
        return person_id

    def add_or_update_person(self, **kwargs) -> ProfileUpdateDecision:
        self.observations.append(kwargs)
        return ProfileUpdateDecision(True, None, "test_update")

    def list_persons(self) -> list:
        return []


class ReIdPipelineScopeTests(unittest.TestCase):
    def run_pipeline(self, detections: list[list[Detection]], draw: bool):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = object.__new__(PersonReIdPipeline)
            pipeline.paths = paths_for(Path(folder))
            pipeline.paths.ensure()
            pipeline.config = PipelineConfig(
                encoder_backend="colorhist", max_frames=0, draw_debug=draw,
                min_crop_width=1, min_crop_height=1, crop_padding=0,
                min_good_frames_before_reid=1, min_embedding_quality=0, min_update_quality=0,
                reid_every_n_frames=1,
            )
            capture = FakeCapture(len(detections))
            capture.frames = [np.full((48, 64, 3), 128, dtype=np.uint8) for _ in detections]
            pipeline._open_capture = lambda _source: capture
            pipeline.tracker = SequenceTracker(detections)
            pipeline.encoder = RecordingEncoder()
            pipeline.store = MemoryStore()
            pipeline.profiles = pipeline.store
            writer = FakeWriter()
            with patch("app.pipeline.orchestrator.cv2.VideoWriter", return_value=writer), patch(
                "app.pipeline.orchestrator.save_crop", return_value=Path(folder) / "snapshot.jpg"
            ):
                result = pipeline.process("fixture.mp4")
            self.assertTrue(capture.released)
            self.assertTrue(writer.released)
            self.assertEqual(result.processed_frames, len(detections))
            return pipeline.encoder.crops, pipeline.store

    def test_overlapping_crops_are_identical_with_and_without_annotations(self) -> None:
        frame = [Detection(1, (2, 5, 35, 44), .9), Detection(2, (20, 5, 55, 44), .9)]
        plain, plain_store = self.run_pipeline([frame], False)
        drawn, drawn_store = self.run_pipeline([frame], True)
        self.assertEqual(len(plain), 2)
        for a, b in zip(plain, drawn):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(
            [o["payload"]["quality_score"] for o in plain_store.observations],
            [o["payload"]["quality_score"] for o in drawn_store.observations],
        )

    def test_new_track_cannot_take_identity_of_known_track_processed_later(self) -> None:
        known = Detection(1, (2, 5, 25, 44), .9)
        newcomer = Detection(2, (30, 5, 55, 44), .9)
        for order in ([newcomer, known], [known, newcomer]):
            with self.subTest(order=[item.track_id for item in order]):
                _, store = self.run_pipeline([[known], order], False)
                assignments = {o["track_id"]: o["person_id"] for o in store.observations}
                self.assertNotEqual(assignments[1], assignments[2])
                self.assertFalse(any("motion" in key for key in store.metadata))
                self.assertFalse(any("motion" in key for o in store.observations for key in o["payload"]))


if __name__ == "__main__":
    unittest.main()
