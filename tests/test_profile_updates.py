"""Exact accumulation, rejected identity updates and OSNet-specific databases."""

from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.config import PipelineConfig
from app.pipeline.orchestrator import PersonReIdPipeline
from app.reid.repository import EmbeddingBatch, IdentityProfile
from app.reid.service import ProfileService
from app.reid.embeddings import checked_embedding
from app.storage.encoder_paths import encoder_identity, paths_for_encoder
from app.storage.models import Detection
from app.storage.vector_store import SQLiteVectorStore
from tests.test_evaluation_artifacts import config_for_test, execute_pipeline
from tests.test_evaluation_scope import paths_for, SequenceTracker
from tests.test_pipeline_resources import FakeCapture, FakeWriter
from tests.test_reid_interfaces import MemoryRepository


def add(service, embedding, **kwargs):
    return service.add_or_update_person("person_000001", np.asarray(embedding), "test", 5, 7,
                                         (2, 5, 25, 44), None, kwargs.pop("snapshot_path", None),
                                         kwargs.pop("payload", None), **kwargs)


class WeightedAccumulationTests(unittest.TestCase):
    def test_same_observations_produce_same_profile_in_every_order(self):
        examples = [(np.array([1., 0.]), .7), (np.array([0., 1.]), .8), (np.array([0., 1.]), .9)]
        expected_sum = sum(vector * weight for vector, weight in examples)
        for order in itertools.permutations(examples):
            repository = MemoryRepository()
            service = ProfileService(repository, min_update_similarity=-1)
            for vector, weight in order:
                add(service, vector, payload={"embedding_weight": weight})
            profile = repository.get_profile("person_000001")
            np.testing.assert_allclose(profile.embedding, checked_embedding(expected_sum), atol=1e-6)
            np.testing.assert_allclose(profile.embedding_sum, expected_sum, atol=1e-12)
            self.assertAlmostEqual(profile.embedding_weight_sum, 2.4)
            self.assertEqual(profile.observations, 3)

    def test_initial_three_crop_batch_retains_all_contributions(self):
        repository = MemoryRepository()
        service = ProfileService(repository, min_update_similarity=-1)
        total = np.array([.7, 1.7])
        batch = EmbeddingBatch(total, 2.4, 3)
        add(service, checked_embedding(total), batch=batch)
        add(service, [1., 0.], payload={"embedding_weight": .65})
        profile = repository.get_profile("person_000001")
        np.testing.assert_allclose(profile.embedding_sum, [1.35, 1.7], atol=1e-12)
        np.testing.assert_allclose(profile.embedding, checked_embedding(np.array([1.35, 1.7])), atol=1e-6)
        self.assertEqual(profile.observations, 4)
        self.assertAlmostEqual(profile.embedding_weight_sum, 3.05)

    def test_sqlite_reopen_preserves_raw_sum_and_continues_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "db.sqlite3"
            first = ProfileService(SQLiteVectorStore(path), min_update_similarity=-1)
            add(first, [1., 0.], payload={"embedding_weight": .7})
            add(first, [0., 1.], payload={"embedding_weight": .8})
            second = ProfileService(SQLiteVectorStore(path), min_update_similarity=-1)
            add(second, [0., 1.], payload={"embedding_weight": .9})
            profile = second.repository.get_profile("person_000001")
            np.testing.assert_allclose(profile.embedding_sum, [.7, 1.7], atol=1e-12)
            self.assertEqual(profile.observations, 3)

    def test_pipeline_records_three_initial_crops_and_one_later_crop(self):
        with tempfile.TemporaryDirectory() as folder:
            box = Detection(7, (2, 5, 25, 44), .9)
            pipeline, result, _, _ = execute_pipeline(paths_for(Path(folder)), [[box]] * 4)
            frames = [json.loads(line) for line in result.predictions_path.read_text().splitlines()]
            qualities = [f["detections"][0]["quality_score"] for f in frames]
            profile = pipeline.store.iter_profiles()[0]
            self.assertEqual(profile.observations, 4)
            self.assertAlmostEqual(profile.embedding_weight_sum, sum(qualities))
            np.testing.assert_allclose(profile.embedding_sum, [sum(qualities), 0.], atol=1e-6)


class PipelineQualityGateTests(unittest.TestCase):
    def run_quality_sequence(self, qualities, *, frame_count=None, **overrides):
        config = replace(config_for_test(), min_embedding_quality=.55,
                         min_update_quality=.65, min_good_frames_before_reid=1)
        config = replace(config, **overrides)
        with tempfile.TemporaryDirectory() as folder:
            box = Detection(7, (2, 5, 25, 44), .9)
            with patch("app.pipeline.orchestrator.crop_quality_score",
                       side_effect=[(quality, {}) for quality in qualities]) as quality_score:
                pipeline, result, _, _ = execute_pipeline(
                    paths_for(Path(folder)), [[box]] * (frame_count or len(qualities)), config=config)
            self.assertEqual(quality_score.call_count, len(qualities))
            frames = [json.loads(line) for line in result.predictions_path.read_text().splitlines()]
            predictions = [frame["detections"][0] for frame in frames]
            return len(pipeline.encoder.crops), pipeline.store.iter_profiles(), predictions, result

    def test_unknown_track_uses_only_candidate_quality_to_fill_initial_buffer(self):
        calls, profiles, predictions, _ = self.run_quality_sequence(
            [.54, .55, .60, .55], min_good_frames_before_reid=3)
        self.assertEqual(calls, 3)
        self.assertEqual([item["state"] for item in predictions], [
            "below_candidate_quality", "waiting_for_initial_observations",
            "waiting_for_initial_observations", "created_identity"])
        self.assertEqual(profiles[0].observations, 3)
        self.assertAlmostEqual(profiles[0].embedding_weight_sum, .55 + .60 + .55)

    def test_initial_candidates_are_collected_from_consecutive_accepted_frames(self):
        box = Detection(7, (2, 5, 25, 44), .9)
        config = replace(config_for_test(), min_good_frames_before_reid=3)
        with tempfile.TemporaryDirectory() as folder:
            pipeline, result, _, _ = execute_pipeline(
                paths_for(Path(folder)), [[box]] * 3, config=config,
            )
            frames = [json.loads(line) for line in result.predictions_path.read_text().splitlines()]
            observations = pipeline.store.iter_profiles()[0].observations

        self.assertEqual(len(pipeline.encoder.crops), 3)
        self.assertEqual(observations, 3)
        self.assertEqual([frame["detections"][0]["state"] for frame in frames], [
            "waiting_for_initial_observations",
            "waiting_for_initial_observations",
            "created_identity",
        ])

    def test_initial_blur_and_border_blur_have_separate_hard_gates(self):
        box = Detection(7, (2, 5, 25, 44), .9)
        config = replace(
            config_for_test(), min_good_frames_before_reid=2,
            min_initial_blur_score=.40, min_border_blur_score=.45,
        )
        quality_results = [
            (.8, {"blur": .39, "edge_cutoff": 1.0}),
            (.8, {"blur": .42, "edge_cutoff": .65}),
            (.8, {"blur": .45, "edge_cutoff": .65}),
            (.8, {"blur": .40, "edge_cutoff": 1.0}),
        ]
        with tempfile.TemporaryDirectory() as folder, patch(
            "app.pipeline.orchestrator.crop_quality_score", side_effect=quality_results,
        ):
            pipeline, result, _, _ = execute_pipeline(
                paths_for(Path(folder)), [[box]] * 4, config=config,
            )
            frames = [json.loads(line) for line in result.predictions_path.read_text().splitlines()]
            manifest = json.loads(result.manifest_path.read_text())

        predictions = [frame["detections"][0] for frame in frames]
        self.assertEqual([item["state"] for item in predictions], [
            "below_initial_blur", "below_border_blur",
            "waiting_for_initial_observations", "created_identity",
        ])
        self.assertEqual([item["initial_candidate_count"] for item in predictions], [0, 0, 1, 2])
        self.assertEqual(len(pipeline.encoder.crops), 2)
        self.assertEqual(manifest["summary"]["below_initial_blur"], 1)
        self.assertEqual(manifest["summary"]["below_border_blur"], 1)

    def test_initial_blur_gates_do_not_reject_updates_of_known_tracks(self):
        box = Detection(7, (2, 5, 25, 44), .9)
        config = replace(
            config_for_test(), min_good_frames_before_reid=1,
            min_initial_blur_score=.40, min_border_blur_score=.45,
        )
        quality_results = [
            (.8, {"blur": .9, "edge_cutoff": 1.0}),
            (.8, {"blur": .1, "edge_cutoff": .65}),
        ]
        with tempfile.TemporaryDirectory() as folder, patch(
            "app.pipeline.orchestrator.crop_quality_score", side_effect=quality_results,
        ):
            pipeline, result, _, _ = execute_pipeline(
                paths_for(Path(folder)), [[box]] * 2, config=config,
            )
            frames = [json.loads(line) for line in result.predictions_path.read_text().splitlines()]

        self.assertEqual([frame["detections"][0]["state"] for frame in frames], [
            "created_identity", "profile_update",
        ])
        self.assertEqual(len(pipeline.encoder.crops), 2)

    def test_initial_aspect_ratio_is_a_hard_gate_with_inclusive_boundary(self):
        box = Detection(7, (2, 5, 25, 44), .9)
        config = replace(
            config_for_test(), min_good_frames_before_reid=2,
            min_initial_aspect_ratio_score=.50,
        )
        quality_results = [
            (.9, {"blur": 1.0, "edge_cutoff": 1.0, "aspect_ratio": .49}),
            (.9, {"blur": 1.0, "edge_cutoff": 1.0, "aspect_ratio": .50}),
            (.9, {"blur": 1.0, "edge_cutoff": 1.0, "aspect_ratio": 1.0}),
        ]
        with tempfile.TemporaryDirectory() as folder, patch(
            "app.pipeline.orchestrator.crop_quality_score", side_effect=quality_results,
        ):
            pipeline, result, _, _ = execute_pipeline(
                paths_for(Path(folder)), [[box]] * 3, config=config,
            )
            frames = [json.loads(line) for line in result.predictions_path.read_text().splitlines()]
            manifest = json.loads(result.manifest_path.read_text())

        predictions = [frame["detections"][0] for frame in frames]
        self.assertEqual([item["state"] for item in predictions], [
            "below_initial_aspect_ratio", "waiting_for_initial_observations", "created_identity",
        ])
        self.assertEqual([item["initial_candidate_count"] for item in predictions], [0, 1, 2])
        self.assertEqual(len(pipeline.encoder.crops), 2)
        self.assertEqual(manifest["summary"]["below_initial_aspect_ratio"], 1)

    def test_known_track_below_update_quality_never_calls_encoder(self):
        calls, profiles, predictions, result = self.run_quality_sequence([.7, .60])
        self.assertEqual(calls, 1)
        self.assertEqual(profiles[0].observations, 1)
        self.assertAlmostEqual(profiles[0].embedding_weight_sum, .7)
        np.testing.assert_allclose(profiles[0].embedding_sum, [.7, 0.])
        rejected = predictions[1]
        self.assertEqual(rejected["state"], "below_update_quality")
        self.assertEqual(rejected["quality_score"], .60)
        self.assertEqual(rejected["person_id"], predictions[0]["person_id"])
        self.assertIsNone(rejected["profile_update_accepted"])
        self.assertIsNone(rejected["update_similarity"])
        self.assertIsNone(rejected["snapshot_frame_index"])
        self.assertIn("Skipped low-quality ReID crops: 1", result.warnings)

    def test_known_track_still_requires_candidate_quality_when_it_is_higher(self):
        calls, profiles, predictions, result = self.run_quality_sequence(
            [.85, .75, .79, .80], min_embedding_quality=.80, min_update_quality=.65)
        self.assertEqual(calls, 2)
        self.assertEqual([item["state"] for item in predictions], [
            "created_identity", "below_candidate_quality", "below_candidate_quality", "profile_update"])
        self.assertEqual(profiles[0].observations, 2)
        self.assertAlmostEqual(profiles[0].embedding_weight_sum, .85 + .80)
        self.assertIn("Skipped low-quality ReID crops: 2", result.warnings)

    def test_known_track_accepts_quality_equal_to_update_threshold(self):
        calls, profiles, predictions, _ = self.run_quality_sequence([.55, .65])
        self.assertEqual(calls, 2)
        self.assertEqual(profiles[0].observations, 2)
        self.assertEqual(predictions[1]["state"], "profile_update")
        self.assertTrue(predictions[1]["profile_update_accepted"])
        self.assertAlmostEqual(predictions[1]["update_similarity"], 1)

    def test_disabled_quality_thresholds_still_allow_initialization_and_updates(self):
        calls, profiles, predictions, _ = self.run_quality_sequence(
            [0., 0.], min_embedding_quality=0, min_update_quality=0)
        self.assertEqual(calls, 2)
        self.assertEqual(profiles[0].observations, 2)
        self.assertAlmostEqual(profiles[0].embedding_weight_sum, .1)
        self.assertEqual(predictions[1]["state"], "profile_update")

    def test_known_track_keeps_global_five_frame_update_schedule_after_rejection(self):
        calls, profiles, predictions, _ = self.run_quality_sequence(
            [.55, .60, .65], frame_count=10, reid_every_n_frames=5)
        self.assertEqual(calls, 2)
        self.assertEqual(profiles[0].observations, 2)
        self.assertEqual(predictions[4]["state"], "below_update_quality")
        self.assertEqual(predictions[9]["state"], "profile_update")
        for index in (1, 2, 3, 5, 6, 7, 8):
            self.assertEqual(predictions[index]["state"], "known_track")
            self.assertIsNone(predictions[index]["quality_score"])

    def test_overlapping_people_never_reach_encoder_or_create_profiles(self):
        first = Detection(1, (2, 5, 35, 44), .9)
        second = Detection(2, (20, 5, 55, 44), .9)
        with tempfile.TemporaryDirectory() as folder:
            pipeline, result, _, _ = execute_pipeline(
                paths_for(Path(folder)), [[first, second]],
                config=replace(config_for_test(), min_good_frames_before_reid=1),
            )
            frame = json.loads(result.predictions_path.read_text().splitlines()[0])
            encoded_crops = len(pipeline.encoder.crops)
            person_count = pipeline.store.count_persons()

        self.assertEqual(encoded_crops, 0)
        self.assertEqual(person_count, 0)
        self.assertEqual([item["state"] for item in frame["detections"]],
                         ["overlapping_person", "overlapping_person"])
        self.assertTrue(all(item["person_overlap_ratio"] > .15 for item in frame["detections"]))
        self.assertIn("Skipped ReID crops due to person overlap or cooldown: 2", result.warnings)

    def test_overlap_starts_cooldown_before_profile_updates_resume(self):
        tracked = Detection(1, (2, 5, 25, 44), .9)
        crossing = Detection(2, (15, 5, 38, 44), .9)
        sequence = [[tracked], [tracked, crossing], [tracked], [tracked], [tracked]]
        config = replace(
            config_for_test(), min_good_frames_before_reid=1,
            max_person_overlap_ratio=.15, overlap_cooldown_frames=2,
        )
        with tempfile.TemporaryDirectory() as folder:
            pipeline, result, _, _ = execute_pipeline(
                paths_for(Path(folder)), sequence, config=config,
            )
            frames = [json.loads(line) for line in result.predictions_path.read_text().splitlines()]
            encoded_crops = len(pipeline.encoder.crops)
            observations = pipeline.store.iter_profiles()[0].observations

        self.assertEqual(encoded_crops, 2)
        self.assertEqual(observations, 2)
        self.assertEqual(frames[1]["detections"][0]["state"], "overlapping_person")
        self.assertEqual(frames[2]["detections"][0]["state"], "overlap_cooldown")
        self.assertEqual(frames[3]["detections"][0]["state"], "overlap_cooldown")
        self.assertEqual(frames[4]["detections"][0]["state"], "profile_update")


class ProfileProtectionTests(unittest.TestCase):
    def test_rejected_update_preserves_all_profile_fields_and_records_reason(self):
        with tempfile.TemporaryDirectory() as folder:
            repository = SQLiteVectorStore(Path(folder) / "db.sqlite3")
            service = ProfileService(repository, min_update_similarity=.82)
            add(service, [1., 0.], snapshot_path="a.jpg", payload={"quality_score": .7})
            before = repository.get_profile("person_000001")
            decision = add(service, [0., 1.], snapshot_path="b.jpg", payload={"quality_score": .99})
            after = repository.get_profile("person_000001")
            self.assertFalse(decision.accepted)
            self.assertEqual(decision.similarity, 0)
            np.testing.assert_array_equal(before.embedding_sum, after.embedding_sum)
            np.testing.assert_array_equal(before.embedding, after.embedding)
            for name in ("observations", "embedding_weight_sum", "created_at", "last_seen",
                         "best_snapshot_path", "best_snapshot_quality"):
                self.assertEqual(getattr(before, name), getattr(after, name))
            event = repository.events_dataframe().iloc[0]
            self.assertEqual(event["event_type"], "profile_update_rejected")
            payload = json.loads(event["payload_json"])
            self.assertEqual(payload["profile_update_reason"], "below_update_similarity")

    def test_invalid_vectors_and_nonfinite_quality_never_mutate_profiles(self):
        repository = MemoryRepository()
        service = ProfileService(repository)
        for vector in ([], [0., 0.], [np.nan, 1.], [np.inf, 1.], [[1., 0.]]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                add(service, vector)
        for quality in (np.nan, np.inf):
            with self.assertRaises(ValueError):
                add(service, [1., 0.], payload={"embedding_weight": quality})
        self.assertEqual(repository.events, [])
        self.assertEqual(repository.profiles, {})

    def test_dimension_change_is_rejected_without_overwriting(self):
        repository = MemoryRepository()
        service = ProfileService(repository)
        add(service, [1., 0.])
        with self.assertRaisesRegex(ValueError, "dimension changed"):
            add(service, [1., 0., 0.])
        self.assertEqual(repository.get_profile("person_000001").observations, 1)
        self.assertEqual(len(repository.events), 1)

    def test_legacy_profile_is_not_falsely_reconstructed(self):
        repository = MemoryRepository()
        previous = IdentityProfile("person_000001", np.array([1., 0.]), 4, 3., "old", "old")
        repository.profiles[previous.person_id] = previous
        with self.assertRaisesRegex(ValueError, "fresh profile database"):
            add(ProfileService(repository), [1., 0.])
        self.assertIs(repository.get_profile(previous.person_id), previous)
        self.assertEqual(repository.events, [])

    def test_simulated_tracker_switch_keeps_profile_but_reports_rejected_update(self):
        class SwitchingEncoder:
            calls = 0
            def encode(self, crop_bgr):
                self.calls += 1
                return np.array([1., 0.]) if self.calls == 1 else np.array([0., 1.])
        with tempfile.TemporaryDirectory() as folder:
            box = Detection(7, (2, 5, 25, 44), .9)
            pipeline = PersonReIdPipeline(replace(config_for_test(), min_good_frames_before_reid=1),
                                         paths_for(Path(folder)), encoder=SwitchingEncoder(),
                                         tracker=SequenceTracker([[box], [box]]))
            pipeline._open_capture = lambda source: FakeCapture(2)
            with patch("app.pipeline.orchestrator.cv2.VideoWriter", return_value=FakeWriter()):
                result = pipeline.process("fixture.mp4")
            profile = pipeline.store.iter_profiles()[0]
            np.testing.assert_array_equal(profile.embedding, [1., 0.])
            self.assertEqual(profile.observations, 1)
            frames = [json.loads(line) for line in result.predictions_path.read_text().splitlines()]
            prediction = frames[1]["detections"][0]
            self.assertEqual(prediction["state"], "profile_update_rejected")
            self.assertFalse(prediction["profile_update_accepted"])
            self.assertEqual(prediction["update_similarity"], 0)
            self.assertIsNotNone(prediction["person_id"])  # No implicit tracker/identity repair.


class EncoderDatabaseTests(unittest.TestCase):
    def test_scoped_paths_are_idempotent_and_switching_does_not_nest_namespaces(self):
        with tempfile.TemporaryDirectory() as folder:
            base = paths_for(Path(folder))
            baseline = config_for_test()
            variant = replace(baseline, reid_model_name="osnet_x0_5",
                              reid_checkpoint="data/models/nonexistent-for-path-test.pth")
            scoped = paths_for_encoder(base, baseline)
            self.assertEqual(scoped, paths_for_encoder(scoped, baseline))
            switched = paths_for_encoder(scoped, variant)
            self.assertEqual(switched, paths_for_encoder(base, variant))
            self.assertEqual(scoped, paths_for_encoder(switched, baseline))

    def test_pipeline_scopes_explicit_base_paths_and_reuses_each_osnet_database(self):
        with tempfile.TemporaryDirectory() as folder:
            base = paths_for(Path(folder))
            baseline = config_for_test()
            variant = replace(baseline, reid_model_name="osnet_x0_5",
                              reid_checkpoint="data/models/nonexistent-for-path-test.pth")
            first = PersonReIdPipeline(baseline, base)
            other = PersonReIdPipeline(variant, base)
            again = PersonReIdPipeline(baseline, base)
            self.assertEqual(first.paths, paths_for_encoder(base, baseline))
            self.assertNotEqual(first.store.db_path, other.store.db_path)
            self.assertEqual(first.store.db_path, again.store.db_path)
            self.assertFalse(base.db_path.exists())

    def test_explicit_repository_keeps_its_path_with_custom_osnet_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            base = paths_for(Path(folder))
            store = SQLiteVectorStore(base.db_path)
            config = replace(config_for_test(), reid_model_name="osnet_x0_5")
            pipeline = PersonReIdPipeline(config, base, store=store)
            self.assertEqual(pipeline.paths, base)
            self.assertIs(pipeline.store, store)

    def test_changing_osnet_model_separates_databases_but_thresholds_do_not(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            baseline = PipelineConfig()
            variant = replace(baseline, reid_model_name="osnet_x0_5")
            self.assertNotEqual(paths_for_encoder(paths, baseline).db_path,
                                paths_for_encoder(paths, variant).db_path)
            self.assertEqual(paths_for_encoder(paths, baseline).db_path,
                             paths_for_encoder(paths, replace(baseline, min_update_quality=.9)).db_path)
            self.assertNotEqual(paths_for_encoder(paths, baseline).db_path, paths.db_path)

    def test_checkpoint_contents_and_architecture_define_namespace_not_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            paths = paths_for(base)
            first, copy = base / "first.pth", base / "copy.pth"
            # Simulate hashing; no model initialization/download is needed here.
            def checksum(path):
                return "same" if Path(path).name in ("first.pth", "copy.pth") else "changed"
            with patch("app.storage.encoder_paths.sha256_file", side_effect=checksum), \
                 patch("pathlib.Path.is_file", return_value=True):
                config = PipelineConfig(reid_checkpoint=str(first))
                a = paths_for_encoder(paths, config).db_path
                self.assertEqual(a, paths_for_encoder(paths, replace(config, reid_checkpoint=str(copy))).db_path)
                self.assertNotEqual(a, paths_for_encoder(paths, replace(config, reid_checkpoint="changed.pth")).db_path)
                self.assertNotEqual(a, paths_for_encoder(paths, replace(config, reid_model_name="osnet_x0_5")).db_path)

    def test_matching_and_quality_thresholds_do_not_change_encoder_identity(self):
        config = PipelineConfig()
        self.assertEqual(
            encoder_identity(config),
            encoder_identity(replace(config, match_threshold=.2, min_update_quality=.9)),
        )


if __name__ == "__main__":
    unittest.main()
