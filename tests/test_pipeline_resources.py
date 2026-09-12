from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.config import AppPaths, PipelineConfig
from app.pipeline.orchestrator import PersonReIdPipeline


class FakeCapture:
    def __init__(self, frame_count: int = 3, opened: bool = True) -> None:
        self.frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(frame_count)]
        self.opened = opened
        self.read_calls = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def get(self, property_id: int) -> float:
        values = {7: len(self.frames), 5: 25.0, 3: 64.0, 4: 48.0}
        return values.get(property_id, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_calls += 1
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class FakeWriter:
    def __init__(self, opened: bool = True) -> None:
        self.opened = opened
        self.frames_written = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def write(self, _frame: np.ndarray) -> None:
        self.frames_written += 1

    def release(self) -> None:
        self.released = True


class FakeStore:
    def add_analysis_run(self, **_kwargs) -> None:
        return None

    def list_persons(self) -> list[object]:
        return []


class EmptyTracker:
    def track_frame(self, _frame: np.ndarray) -> list[object]:
        return []


class FailingTracker:
    def track_frame(self, _frame: np.ndarray) -> list[object]:
        raise RuntimeError("tracker failed")


class PipelineResourceTests(unittest.TestCase):
    def make_pipeline(self, temp_dir: str, capture: FakeCapture, tracker: object) -> PersonReIdPipeline:
        pipeline = object.__new__(PersonReIdPipeline)
        pipeline.config = PipelineConfig(
            encoder_backend="colorhist",
            max_frames=2,
            draw_debug=False,
        )
        base = Path(temp_dir)
        pipeline.paths = AppPaths(
            db_path=base / "db.sqlite3",
            snapshot_dir=base / "snapshots",
            output_dir=base / "output",
            mode_dir=base / "modes",
            input_dir=base / "input",
        )
        pipeline.paths.ensure()
        pipeline.store = FakeStore()
        pipeline.tracker = tracker
        pipeline._open_capture = lambda _source: capture
        return pipeline

    def test_invalid_writer_fails_and_releases_all_resources(self) -> None:
        capture = FakeCapture()
        writer = FakeWriter(opened=False)
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "app.pipeline.orchestrator.cv2.VideoWriter", return_value=writer
        ):
            pipeline = self.make_pipeline(temp_dir, capture, EmptyTracker())
            with self.assertRaisesRegex(RuntimeError, "output video writer"):
                pipeline.process("video.mp4")

        self.assertTrue(capture.released)
        self.assertTrue(writer.released)

    def test_processing_error_releases_capture_and_writer(self) -> None:
        capture = FakeCapture()
        writer = FakeWriter()
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "app.pipeline.orchestrator.cv2.VideoWriter", return_value=writer
        ):
            pipeline = self.make_pipeline(temp_dir, capture, FailingTracker())
            with self.assertRaisesRegex(RuntimeError, "tracker failed"):
                pipeline.process("video.mp4")

        self.assertTrue(capture.released)
        self.assertTrue(writer.released)

    def test_frame_limit_does_not_read_or_report_an_extra_frame(self) -> None:
        capture = FakeCapture(frame_count=3)
        writer = FakeWriter()
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "app.pipeline.orchestrator.cv2.VideoWriter", return_value=writer
        ):
            pipeline = self.make_pipeline(temp_dir, capture, EmptyTracker())
            result = pipeline.process("video.mp4")

        self.assertEqual(result.processed_frames, 2)
        self.assertEqual(capture.read_calls, 2)
        self.assertEqual(writer.frames_written, 2)
        self.assertTrue(capture.released)
        self.assertTrue(writer.released)


if __name__ == "__main__":
    unittest.main()
