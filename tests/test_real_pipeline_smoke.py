"""Opt-in tests with real models and generated, non-personal video fixtures.

Run with REID_REAL_SMOKE=1 after installing dependencies/downloading the
documented weights. These tests measure plumbing, not recognition accuracy.
"""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from app.config import PROJECT_ROOT
from app.evaluation.runner import run_unit
from app.modes.mode_registry import builtin_modes
from app.pipeline.model_weights import DEFAULT_CHECKPOINT_SHA256
from app.pipeline.reid_encoder import TorchreidOSNetEncoder
from tests.test_evaluation_scope import paths_for


@unittest.skipUnless(os.getenv("REID_REAL_SMOKE") == "1", "Opt-in: set REID_REAL_SMOKE=1")
class RealPipelineSmokeTests(unittest.TestCase):
    def test_real_osnet_produces_repeatable_normalized_embeddings(self):
        encoder = TorchreidOSNetEncoder(device="cpu", checkpoint_path="data/models/osnet_x1_0_msmt17.pth")
        crop = np.full((256, 128, 3), 128, np.uint8)
        first, second = encoder.encode(crop), encoder.encode(crop)
        self.assertEqual(first.shape, (512,))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.allclose(first, second))
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1, places=5)
        self.assertEqual(encoder.checkpoint_sha256, DEFAULT_CHECKPOINT_SHA256)

    def test_real_capture_yolo_encoder_database_writer_and_exports_for_all_presets(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "blank.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(4): writer.write(np.full((48, 64, 3), 128, np.uint8))
            finally:
                writer.release()
            for mode in builtin_modes().values():
                with self.subTest(mode=mode.mode_id):
                    config = replace(mode.to_pipeline_config(), max_frames=0, device="cpu",
                                     yolo_model=str(PROJECT_ROOT / "yolov8n.pt"))
                    unit_path = run_unit([source], config, root=base / "experiments", base_paths=paths_for(base))
                    unit = json.loads(unit_path.read_text())
                    run = json.loads(Path(unit["runs"][0]["manifest"]).read_text())
                    self.assertEqual(unit["status"], "completed")
                    self.assertEqual(run["status"], "completed")
                    self.assertEqual(run["processed_frames"], 4)
                    frames = [json.loads(line) for line in Path(run["exports"]["frames_jsonl"]).read_text().splitlines()]
                    self.assertEqual(len(frames), 4)
                    self.assertTrue(all(not frame["detections"] for frame in frames))
                    self.assertTrue(run["input"]["sha256"])
                    self.assertTrue(run["models"]["detector"]["sha256"])
                    self.assertTrue(run["models"]["tracker"]["sha256"])
                    self.assertEqual(run["models"]["encoder"]["checkpoint"]["sha256"], DEFAULT_CHECKPOINT_SHA256)
                    capture = cv2.VideoCapture(run["exports"]["annotated_video"])
                    try:
                        self.assertTrue(capture.isOpened())
                        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
                    finally:
                        capture.release()
