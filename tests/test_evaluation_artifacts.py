from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from app.config import PipelineConfig
from app.evaluation.artifacts import sha256_file
from app.evaluation.runner import run_unit
from app.pipeline.detector_tracker import UltralyticsPersonTracker
from app.pipeline.orchestrator import PersonReIdPipeline
from app.pipeline.reid_encoder import TorchreidOSNetEncoder
from app.storage.models import Detection
from app.utils.image_utils import save_crop
from tests.test_evaluation_scope import paths_for, SequenceTracker, RecordingEncoder
from tests.test_pipeline_resources import FakeCapture, FakeWriter


def config_for_test():
    return PipelineConfig(encoder_backend="colorhist", max_frames=0, draw_debug=False,
                          min_crop_width=1, min_crop_height=1, crop_padding=0,
                          min_good_frames_before_reid=3, min_embedding_quality=0,
                          min_update_quality=0, reid_every_n_frames=1)


def execute_pipeline(paths, detections, config=None, source="fixture.mp4"):
    pipeline = PersonReIdPipeline(config or config_for_test(), paths,
                                 tracker=SequenceTracker(detections), encoder=RecordingEncoder())
    capture, writer = FakeCapture(len(detections)), FakeWriter()
    pipeline._open_capture = lambda source: capture
    with patch("app.pipeline.orchestrator.cv2.VideoWriter", return_value=writer):
        result = pipeline.process(source)
    return pipeline, result, capture, writer


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def __getitem__(self, index): return Tensor(self.value[index])
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value
    def item(self): return self.value.item()


class EvaluationArtifactTests(unittest.TestCase):
    def test_missing_tracker_ids_are_not_fabricated(self):
        tracker = object.__new__(UltralyticsPersonTracker)
        tracker.tracker, tracker.confidence, tracker.image_size, tracker.device = "bytetrack.yaml", .35, 640, None
        box = SimpleNamespace(xyxy=Tensor([[2, 5, 25, 44]]), conf=Tensor([.9]), cls=Tensor([0]), id=None)
        tracker.model = SimpleNamespace(track=lambda *args, **kwargs: [SimpleNamespace(boxes=[box])])
        first = tracker.track_frame(np.zeros((48, 64, 3), dtype=np.uint8))
        box.id = Tensor([1])
        second = tracker.track_frame(np.zeros((48, 64, 3), dtype=np.uint8))
        self.assertIsNone(first[0].track_id)
        self.assertEqual(second[0].track_id, 1)

    def test_complete_export_and_decision_frame_without_backfill(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            tracked = Detection(1, (2, 5, 25, 44), .9)
            untracked = Detection(None, (2, 5, 25, 44), .9)
            pipeline, result, capture, writer = execute_pipeline(paths, [[untracked], [tracked], [tracked], [tracked], []])
            frames = [json.loads(line) for line in result.predictions_path.read_text().splitlines()]
            self.assertEqual([frame["frame_index"] for frame in frames], [1, 2, 3, 4, 5])
            self.assertEqual(frames[-1]["detections"], [])
            self.assertIsNone(frames[0]["detections"][0]["track_id"])
            self.assertEqual(frames[0]["detections"][0]["state"], "untracked")
            self.assertIsNone(frames[1]["detections"][0]["person_id"])
            self.assertIsNone(frames[2]["detections"][0]["person_id"])
            decision = frames[3]["detections"][0]
            self.assertEqual(decision["decision_frame_index"], 4)
            self.assertEqual(decision["snapshot_frame_index"], 2)
            event = pipeline.store.events_dataframe().iloc[0]
            self.assertEqual(event["frame_index"], 4)
            payload = json.loads(event["payload_json"])
            self.assertEqual(payload["snapshot_frame_index"], 2)
            self.assertEqual(payload["decision_frame_index"], 4)
            self.assertIn(result.run_id, event["snapshot_path"])
            self.assertTrue(Path(event["snapshot_path"]).is_file())
            mot = result.tracking_predictions_path.read_text().splitlines()
            self.assertEqual([int(line.split(",")[0]) for line in mot], [2, 3, 4])
            self.assertTrue(all(len(line.split(",")) == 10 for line in mot))
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["processed_frames"], 5)
            self.assertEqual(manifest["exports"]["sha256"]["frames_jsonl"], sha256_file(result.predictions_path))
            self.assertIsNone(manifest["models"]["encoder"]["checkpoint"])
            self.assertIn("source_tree_sha256", manifest["code"])
            self.assertGreater(manifest["timings"]["processing_seconds"], 0)
            run = pipeline.store.analysis_runs_dataframe().iloc[0]
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["processed_frames"], 5)
            self.assertTrue(capture.released and writer.released)

    def test_untracked_boxes_never_build_a_person_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            box = Detection(None, (2, 5, 25, 44), .9)
            pipeline, result, _, _ = execute_pipeline(paths_for(Path(folder)), [[box]] * 4)
            self.assertEqual(result.created_persons, 0)
            self.assertEqual(pipeline.store.count_persons(), 0)
            self.assertEqual(result.tracking_predictions_path.read_text(), "")

    def test_failure_manifest_keeps_completed_frames_and_original_error(self):
        class FailingOnSecondFrame:
            calls = 0
            def track_frame(self, frame):
                self.calls += 1
                if self.calls == 2: raise RuntimeError("second frame failed")
                return []
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            pipeline = PersonReIdPipeline(config_for_test(), paths, tracker=FailingOnSecondFrame(), encoder=RecordingEncoder())
            capture, writer = FakeCapture(3), FakeWriter()
            pipeline._open_capture = lambda source: capture
            with patch("app.pipeline.orchestrator.cv2.VideoWriter", return_value=writer):
                with self.assertRaisesRegex(RuntimeError, "second frame failed"):
                    pipeline.process("fixture.mp4")
            manifest = json.loads(pipeline.last_manifest_path.read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["processed_frames"], 1)
            self.assertEqual(manifest["error"]["message"], "second frame failed")
            self.assertEqual(pipeline.store.analysis_runs_dataframe().iloc[0]["status"], "failed")
            self.assertTrue(capture.released and writer.released)

    def test_model_start_failure_is_logged_before_capture_is_opened(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            config = replace(config_for_test(), encoder_backend="torchreid", reid_checkpoint=str(Path(folder) / "missing.pth"))
            pipeline = PersonReIdPipeline(config, paths, tracker=SequenceTracker([]))
            with patch.object(pipeline, "_open_capture") as open_capture:
                with self.assertRaises(FileNotFoundError): pipeline.process("fixture.mp4")
            open_capture.assert_not_called()
            manifest = json.loads(pipeline.last_manifest_path.read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["processed_frames"], 0)
            self.assertEqual(pipeline.store.analysis_runs_dataframe().iloc[0]["status"], "failed")

    def test_pipeline_cannot_reuse_persistent_tracker_state(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline, result, _, _ = execute_pipeline(paths_for(Path(folder)), [[]])
            with self.assertRaisesRegex(RuntimeError, "one source only"):
                pipeline.process("second.mp4")
            self.assertEqual(len(list((pipeline.paths.output_dir / "runs").iterdir())), 1)

    def test_database_finalization_failure_cannot_leave_a_success_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = PersonReIdPipeline(config_for_test(), paths_for(Path(folder)),
                                         tracker=SequenceTracker([[]]), encoder=RecordingEncoder())
            pipeline._open_capture = lambda source: FakeCapture(1)
            with patch("app.pipeline.orchestrator.cv2.VideoWriter", return_value=FakeWriter()), patch.object(
                pipeline.store, "finish_analysis_run", side_effect=RuntimeError("final DB write failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "final DB write failed"):
                    pipeline.process("fixture.mp4")
            manifest = json.loads(pipeline.last_manifest_path.read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["error"]["message"], "final DB write failed")

    def test_file_video_ending_early_is_marked_failed(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "incomplete.mp4"
            source.write_bytes(b"opened through FakeCapture")
            pipeline = PersonReIdPipeline(config_for_test(), paths_for(base),
                                         tracker=SequenceTracker([[]]), encoder=RecordingEncoder())
            capture = FakeCapture(3)
            capture.frames = capture.frames[:1]
            original_get = capture.get
            capture.get = lambda property_id: 3 if property_id == cv2.CAP_PROP_FRAME_COUNT else original_get(property_id)
            pipeline._open_capture = lambda source: capture
            with patch("app.pipeline.orchestrator.cv2.VideoWriter", return_value=FakeWriter()):
                with self.assertRaisesRegex(RuntimeError, "Video ended early"):
                    pipeline.process(str(source))
            manifest = json.loads(pipeline.last_manifest_path.read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["processed_frames"], 1)

    def test_no_decodable_frames_is_a_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            pipeline = PersonReIdPipeline(config_for_test(), paths_for(Path(folder)),
                                         tracker=SequenceTracker([]), encoder=RecordingEncoder())
            pipeline._open_capture = lambda source: FakeCapture(0)
            with patch("app.pipeline.orchestrator.cv2.VideoWriter", return_value=FakeWriter()):
                with self.assertRaisesRegex(RuntimeError, "no decodable frames"):
                    pipeline.process("fixture.mp4")
            self.assertEqual(json.loads(pipeline.last_manifest_path.read_text())["status"], "failed")

    def test_units_isolate_databases_but_share_profiles_between_related_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            sources = [base / "register.mp4", base / "return.mp4"]
            for source in sources: source.write_bytes(b"test fixture, opened through FakeCapture")
            created = []
            initial_counts = []
            def factory(config, paths):
                box = Detection(1, (2, 5, 25, 44), .9)
                pipeline = PersonReIdPipeline(config, paths, tracker=SequenceTracker([[box]]), encoder=RecordingEncoder())
                initial_counts.append(pipeline.store.count_persons())
                pipeline._open_capture = lambda source: FakeCapture(1)
                created.append(pipeline)
                return pipeline
            config = replace(config_for_test(), min_good_frames_before_reid=1)
            with patch("app.pipeline.orchestrator.cv2.VideoWriter", side_effect=lambda *args: FakeWriter()):
                first = run_unit(sources, config, root=base / "experiments", base_paths=paths_for(base), pipeline_factory=factory)
                second = run_unit(sources[:1], config, root=base / "experiments", base_paths=paths_for(base), pipeline_factory=factory)
            self.assertEqual(initial_counts, [0, 1, 0])
            self.assertEqual(created[0].paths.db_path, created[1].paths.db_path)
            self.assertNotEqual(created[0].paths.db_path, created[2].paths.db_path)
            self.assertIsNot(created[0].tracker, created[1].tracker)
            self.assertEqual(created[1].store.count_persons(), 1)
            self.assertEqual(json.loads(first.read_text())["status"], "completed")
            self.assertNotEqual(first, second)

    def test_failed_unit_records_its_failed_source_and_run_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "source.mp4"
            source.write_bytes(b"fixture")
            def factory(config, paths):
                pipeline = PersonReIdPipeline(config, paths, tracker=SequenceTracker([]))
                return pipeline
            config = replace(config_for_test(), encoder_backend="torchreid", reid_checkpoint=str(base / "missing.pth"))
            with self.assertRaises(FileNotFoundError):
                run_unit([source], config, root=base / "experiments", base_paths=paths_for(base), pipeline_factory=factory)
            manifest = json.loads(next((base / "experiments").glob("*/experiment.json")).read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["runs"][0]["status"], "failed")
            self.assertTrue(Path(manifest["runs"][0]["manifest"]).is_file())


class SnapshotAndCheckpointTests(unittest.TestCase):
    def test_zero_detection_confidence_is_rejected_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "Ultralytics replaces zero"):
            UltralyticsPersonTracker(confidence=0)

    def test_partial_or_unloaded_backbone_checkpoints_are_rejected(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "test.pth"
            expected = {"conv.weight": torch.ones(2), "conv.bias": torch.zeros(2)}
            extractor = SimpleNamespace(model=SimpleNamespace(state_dict=lambda: expected, feature_dim=2))
            module = SimpleNamespace(FeatureExtractor=lambda **kwargs: extractor)
            for state, message in (
                ({"conv.weight": torch.ones(2)}, "missing/mismatched backbone"),
                ({"conv.weight": torch.zeros(2), "conv.bias": torch.zeros(2)}, "did not load all"),
            ):
                torch.save(state, checkpoint)
                with patch("app.pipeline.reid_encoder.importlib.import_module", return_value=module):
                    with self.assertRaisesRegex(ValueError, message):
                        TorchreidOSNetEncoder(checkpoint_path=str(checkpoint))

    def test_snapshots_from_different_runs_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            first = save_crop(np.zeros((10, 10, 3), np.uint8), base, "person_000001", 1, run_id="first")
            second = save_crop(np.full((10, 10, 3), 255, np.uint8), base, "person_000001", 1, run_id="second")
            self.assertNotEqual(first, second)
            self.assertEqual(int(cv2.imread(str(first)).mean()), 0)
            self.assertEqual(int(cv2.imread(str(second)).mean()), 255)

    def test_snapshot_write_failure_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as folder, patch("app.utils.image_utils.cv2.imwrite", return_value=False):
            with self.assertRaises(OSError):
                save_crop(np.zeros((10, 10, 3), np.uint8), Path(folder), "person_000001", 1, run_id="run")

    def test_snapshot_identifiers_cannot_escape_the_run_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            for run_id, person_id in (("../outside", "person_1"), ("run", "../outside")):
                with self.assertRaises(ValueError):
                    save_crop(np.zeros((10, 10, 3), np.uint8), Path(folder), person_id, 1, run_id=run_id)

    def test_osnet_never_falls_back_when_checkpoint_is_missing(self):
        with self.assertRaisesRegex(ValueError, "explicit ReID checkpoint"):
            TorchreidOSNetEncoder(checkpoint_path="")
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                TorchreidOSNetEncoder(checkpoint_path=str(Path(folder) / "missing.pth"))
