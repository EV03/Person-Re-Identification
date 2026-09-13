"""Contract and behavior regressions for backend-independent ReID composition."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.config import PipelineConfig, PipelineSettings
from app.modes.base_mode import ModeConfig
from app.pipeline.orchestrator import PersonReIdPipeline
from app.reid.repository import IdentityProfile, ProfileObservation
from app.reid.operations import WeightedMeanProfileUpdater
from app.reid.service import ProfileService
from app.reid.embeddings import normalize_vector
from app.storage.models import Detection, PersonRecord
from app.storage.vector_store import SQLiteVectorStore
from tests.test_evaluation_artifacts import config_for_test
from tests.test_evaluation_scope import RecordingEncoder, paths_for
from tests.test_pipeline_resources import FakeCapture, FakeWriter


class MemoryRepository:
    """Only persistence: deliberately no search/update methods or SQL."""

    def __init__(self):
        self.profiles: dict[str, IdentityProfile] = {}
        self.events: list[ProfileObservation] = []

    def get_profile(self, person_id):
        return self.profiles.get(person_id)

    def iter_profiles(self):
        return list(self.profiles.values())

    def save_observation(self, profile, observation):
        self.profiles[profile.person_id] = profile
        self.events.append(observation)

    def create_person_id(self):
        return f"person_{len(self.profiles) + 1:06d}"

    def list_persons(self):
        return [PersonRecord(p.person_id, p.observations, p.created_at, p.last_seen,
                             p.best_snapshot_path) for p in self.profiles.values()]


def observe(service, vector, person_id="person_000001", **payload):
    service.add_or_update_person(person_id, np.asarray(vector), "fixture", 1, 7,
                                 (0, 0, 10, 20), None, None, payload)


class SettingsContractTests(unittest.TestCase):
    def test_settings_are_inherited_not_redeclared(self):
        mode = ModeConfig("pilot", "Pilot", "test")
        config = PipelineConfig()
        for field in fields(PipelineSettings):
            with self.subTest(field=field.name):
                self.assertIs(ModeConfig.__dataclass_fields__[field.name], field)
                self.assertIs(PipelineConfig.__dataclass_fields__[field.name], field)
                self.assertEqual(getattr(mode, field.name), getattr(config, field.name))

    def test_all_shared_values_survive_preset_json_and_runtime_conversion(self):
        mode = ModeConfig("pilot", "Pilot", "test", min_update_quality=.713,
                          tracker="botsort.yaml", match_threshold=.812)
        restored = ModeConfig.from_json_dict(mode.to_json_dict())
        config = restored.to_pipeline_config()
        for field in fields(PipelineSettings):
            self.assertEqual(getattr(config, field.name), getattr(mode, field.name))
        self.assertEqual(config.mode_name, "Pilot")
        self.assertEqual(replace(config, max_frames=0).max_frames, 0)


class ProfileContractTests(unittest.TestCase):
    def test_minimum_matching_threshold_includes_opposite_valid_vector(self):
        repository = MemoryRepository()
        service = ProfileService(repository)
        observe(service, [1., 0.])
        match = service.search(np.array([-1., 0.]), threshold=-1)
        self.assertIsNotNone(match)
        self.assertEqual(match.person_id, "person_000001")
        self.assertEqual(match.score, -1)

    def test_same_service_works_with_memory_and_sqlite(self):
        with tempfile.TemporaryDirectory() as folder:
            for repository in (MemoryRepository(), SQLiteVectorStore(Path(folder) / "db.sqlite3")):
                with self.subTest(repository=type(repository).__name__):
                    service = ProfileService(repository)
                    observe(service, [1., 0.])
                    observe(service, [0., 1.], person_id="person_000002")
                    self.assertEqual(service.search(np.array([1., 0.]), .8).person_id, "person_000001")
                    self.assertIsNone(service.search(np.array([1., 0.]), .8, {"person_000001"}))
                    self.assertIsNone(service.search(np.ones(3), .5))
                    self.assertEqual(len(service.list_persons()), 2)

    def test_policies_are_replaceable_without_repository_changes(self):
        class NeverMatch:
            def match(self, embedding, profiles, threshold, exclude_person_ids):
                return None

        class LatestEmbedding:
            def update(self, previous, **kwargs):
                return IdentityProfile(kwargs["person_id"], normalize_vector(kwargs["embedding"]),
                                       1, kwargs["weight"], kwargs["timestamp"], kwargs["timestamp"])

        repository = MemoryRepository()
        service = ProfileService(repository, matcher=NeverMatch(), updater=LatestEmbedding(), min_update_similarity=-1)
        observe(service, [1., 0.])
        observe(service, [0., 1.])
        np.testing.assert_array_equal(repository.get_profile("person_000001").embedding, [0., 1.])
        self.assertIsNone(service.search(np.array([0., 1.]), .1))
        self.assertEqual(len(repository.events), 2)

    def test_update_accumulates_exact_weighted_mean(self):
        repository = MemoryRepository()
        service = ProfileService(repository, min_update_similarity=-1)
        weighted_sum = np.zeros(2)
        total = 0.
        for vector, weight in (([1., 0.], .8), ([0., 1.], .5), ([1., 1.], .7)):
            normalized = normalize_vector(np.array(vector))
            weighted_sum += normalized * weight
            total += weight
            observe(service, vector, embedding_weight=weight)
        profile = repository.get_profile("person_000001")
        np.testing.assert_allclose(profile.embedding, normalize_vector(weighted_sum), atol=1e-6)
        np.testing.assert_allclose(profile.embedding_sum, weighted_sum, atol=1e-6)
        self.assertEqual(profile.observations, 3)
        self.assertAlmostEqual(profile.embedding_weight_sum, total)

    def test_snapshot_quality_is_preserved_and_dimension_change_is_rejected(self):
        policy = WeightedMeanProfileUpdater()
        previous = IdentityProfile("id", np.array([1., 0.]), 3, 2., "old", "old", "best.jpg", .9, np.array([2., 0.]))
        updated = policy.update(previous, person_id="id", embedding=np.ones(2), weight=.4,
                                snapshot_path="worse.jpg", snapshot_quality=.2, timestamp="new")
        self.assertEqual(updated.embedding.shape, (2,))
        self.assertEqual(updated.embedding_weight_sum, 2.4)
        self.assertEqual(updated.best_snapshot_path, "best.jpg")
        self.assertEqual(updated.created_at, "old")
        with self.assertRaisesRegex(ValueError, "dimension changed"):
            policy.update(previous, person_id="id", embedding=np.ones(3), weight=.4,
                          snapshot_path=None, snapshot_quality=0, timestamp="new")

    def test_profile_and_event_save_are_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            repository = SQLiteVectorStore(Path(folder) / "db.sqlite3")
            service = ProfileService(repository)
            with self.assertRaises(TypeError):
                observe(service, [1., 0.], unserializable=object())
            self.assertEqual(repository.count_persons(), 0)
            self.assertTrue(repository.events_dataframe().empty)


class TrackerCompositionTests(unittest.TestCase):
    def test_tracker_is_released_when_encoder_loading_fails(self):
        class HandleTracker:
            released = False
            def track_frame(self, frame_bgr): return []
            def release(self): self.released = True
        with tempfile.TemporaryDirectory() as folder:
            tracker = HandleTracker()
            pipeline = PersonReIdPipeline(config_for_test(), paths_for(Path(folder)), tracker=tracker)
            with patch("app.pipeline.orchestrator.build_encoder", side_effect=RuntimeError("encoder loading failed")):
                with self.assertRaisesRegex(RuntimeError, "encoder loading failed"):
                    pipeline.process("fixture.mp4")
            self.assertTrue(tracker.released)
            manifest = json.loads(pipeline.last_manifest_path.read_text())
            self.assertEqual(manifest["status"], "failed")

    def test_separate_profile_repository_does_not_need_run_or_sql_methods(self):
        from tests.test_evaluation_scope import SequenceTracker
        with tempfile.TemporaryDirectory() as folder:
            repository = MemoryRepository()
            profiles = ProfileService(repository)
            pipeline = PersonReIdPipeline(replace(config_for_test(), min_good_frames_before_reid=1),
                                         paths_for(Path(folder)), profiles=profiles, encoder=RecordingEncoder(),
                                         tracker=SequenceTracker([[Detection(1, (2, 5, 25, 44), .9)]]))
            pipeline._open_capture = lambda source: FakeCapture(1)
            with patch("app.pipeline.orchestrator.cv2.VideoWriter", return_value=FakeWriter()):
                result = pipeline.process("fixture.mp4")
            self.assertEqual(len(result.persons), 1)
            self.assertEqual(len(repository.events), 1)
            self.assertEqual(pipeline.store.count_persons(), 0)
            self.assertEqual(pipeline.store.analysis_runs_dataframe().iloc[0]["status"], "completed")

    def test_custom_tracker_factory_has_no_yolo_dependency_and_releases_backend(self):
        class OtherTracker:
            def __init__(self): self.released = False
            def track_frame(self, frame_bgr): return [Detection(42, (2, 5, 25, 44), .9)]
            def release(self): self.released = True
            def describe_backend(self): return {"name": "independent-test-tracker"}

        with tempfile.TemporaryDirectory() as folder:
            tracker, factory_calls = OtherTracker(), []
            def factory(config):
                factory_calls.append(config)
                return tracker
            paths = paths_for(Path(folder))
            pipeline = PersonReIdPipeline(replace(config_for_test(), min_good_frames_before_reid=1), paths,
                                         tracker_factory=factory, encoder=RecordingEncoder())
            pipeline._open_capture = lambda source: FakeCapture(2)
            with patch("app.pipeline.detector_tracker.UltralyticsPersonTracker", side_effect=AssertionError("YOLO loaded")), \
                 patch("app.pipeline.orchestrator.cv2.VideoWriter", return_value=FakeWriter()), \
                 patch.object(pipeline.store, "search", side_effect=AssertionError("DB matching used")), \
                 patch.object(pipeline.store, "add_or_update_person", side_effect=AssertionError("DB update used")):
                result = pipeline.process("fixture.mp4")
            self.assertEqual(len(factory_calls), 1)
            self.assertTrue(tracker.released)
            self.assertEqual(result.created_persons, 1)
            manifest = json.loads(result.manifest_path.read_text())
            self.assertEqual(manifest["components"]["tracker"]["metadata"]["name"], "independent-test-tracker")
            self.assertIsNone(manifest["models"]["detector"]["sha256"])
            self.assertIn("configured_models", manifest)
            self.assertIn("CosineProfileMatcher", manifest["components"]["profiles"]["metadata"]["matcher"])

    def test_custom_matching_is_used_by_pipeline(self):
        class NeverMatch:
            def match(self, embedding, profiles, threshold, exclude_person_ids):
                return None
        from tests.test_evaluation_scope import SequenceTracker
        with tempfile.TemporaryDirectory() as folder:
            pipeline = PersonReIdPipeline(replace(config_for_test(), min_good_frames_before_reid=1),
                                         paths_for(Path(folder)), matcher=NeverMatch(), encoder=RecordingEncoder(),
                                         tracker=SequenceTracker([[Detection(1, (2, 5, 25, 44), .9)],
                                                                  [Detection(2, (2, 5, 25, 44), .9)]]))
            pipeline._open_capture = lambda source: FakeCapture(2)
            with patch("app.pipeline.orchestrator.cv2.VideoWriter", return_value=FakeWriter()):
                result = pipeline.process("fixture.mp4")
            self.assertEqual(result.created_persons, 2)


if __name__ == "__main__":
    unittest.main()
