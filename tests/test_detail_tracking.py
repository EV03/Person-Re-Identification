from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.evaluation.single_person import create_single_person_report, load_prediction_events
from app.config import PipelineConfig
from app.pipeline.detail_tracking import DetailTrackingPolicy, has_sufficient_new_person_evidence
from app.pipeline.orchestrator import PersonReIdPipeline
from app.reid.repository import IdentityProfile
from app.reid.service import ProfileService
from app.storage.vector_store import SQLiteVectorStore
from app.storage.models import Detection
from app.utils.detail_utils import DETAIL_VECTOR_LENGTH, detail_similarity_breakdown, extract_detail_snapshot
from tests.test_evaluation_scope import RecordingEncoder, SequenceTracker, paths_for
from tests.test_pipeline_resources import FakeCapture, FakeWriter


def profile(person_id: str, embedding: list[float], detail: np.ndarray | None) -> IdentityProfile:
    vector = np.asarray(embedding, dtype=np.float64)
    return IdentityProfile(person_id, vector, 1, 1.0, "now", "now",
                           embedding_sum=vector.copy(), detail_vector=detail,
                           detail_weight_sum=1.0 if detail is not None else 0.0)


class DetailFeatureTests(unittest.TestCase):
    def test_snapshot_has_stable_registry_length_and_explainable_payload(self) -> None:
        crop = np.zeros((180, 80, 3), dtype=np.uint8)
        crop[50:120, :] = (20, 180, 240)
        snapshot = extract_detail_snapshot(crop)
        self.assertIsNotNone(snapshot)
        self.assertEqual(len(snapshot.vector), DETAIL_VECTOR_LENGTH)
        self.assertIn("upper_color_hist", snapshot.to_payload()["features"])

    def test_identical_details_support_but_missing_details_are_neutral(self) -> None:
        detail = np.zeros(DETAIL_VECTOR_LENGTH, dtype=np.float32)
        detail[14] = 1.0
        self.assertGreater(detail_similarity_breakdown(detail, detail)["score"], .5)
        self.assertEqual(detail_similarity_breakdown(detail, None)["score"], .5)


class DetailDecisionPolicyTests(unittest.TestCase):
    def test_detail_reranking_and_motion_bonus_are_bounded_support_signals(self) -> None:
        policy = DetailTrackingPolicy(detail_weight=.10, motion_identity_bonus=.04)
        query_detail = np.zeros(DETAIL_VECTOR_LENGTH, dtype=np.float32)
        query_detail[14] = 1.0
        other_detail = np.zeros(DETAIL_VECTOR_LENGTH, dtype=np.float32)
        other_detail[15] = 1.0
        profiles = [profile("matching_detail", [1, 0], query_detail),
                    profile("other_detail", [1, 0], other_detail)]
        matches = policy.rank_profiles(
            np.asarray([1, 0], dtype=np.float32), query_detail, profiles,
            exclude_person_ids=set(), bbox_xyxy=(10, 10, 30, 60), frame_index=10,
            image_width=100, image_height=100, recent_person_positions={},
        )
        self.assertEqual(matches[0].person_id, "matching_detail")
        self.assertLessEqual(matches[0].score, 1.0)
        self.assertEqual(matches[0].visual_score, 1.0)

        motion_matches = policy.rank_profiles(
            np.asarray([1, 0], dtype=np.float32), None, profiles,
            exclude_person_ids=set(), bbox_xyxy=(10, 10, 30, 60), frame_index=10,
            image_width=100, image_height=100,
            recent_person_positions={"other_detail": (9, (20.0, 35.0))},
        )
        boosted = next(match for match in motion_matches if match.person_id == "other_detail")
        self.assertEqual(boosted.motion_bonus, .04)

    def test_new_person_needs_repeated_low_evidence_and_frame_span(self) -> None:
        evidence = [(1, .2), (4, .3), (7, .2)]
        self.assertTrue(has_sufficient_new_person_evidence(
            evidence, max_score=.4, min_events=3, min_span_frames=7, low_match_ratio=.8))
        self.assertFalse(has_sufficient_new_person_evidence(
            evidence[:2], max_score=.4, min_events=3, min_span_frames=7, low_match_ratio=.8))

    def test_ablation_switches_remove_only_the_selected_support_signal(self) -> None:
        detail = np.zeros(DETAIL_VECTOR_LENGTH, dtype=np.float32)
        detail[14] = 1.0
        candidate = profile("person", [1, 0], detail)
        no_detail = DetailTrackingPolicy(detail_reranking_enabled=False).rank_profiles(
            np.asarray([1, 0], dtype=np.float32), detail, [candidate],
            exclude_person_ids=set(), bbox_xyxy=(10, 10, 30, 60), frame_index=10,
            image_width=100, image_height=100, recent_person_positions={},
        )[0]
        self.assertIsNone(no_detail.detail_score)
        self.assertEqual(no_detail.detail_weight, 0)
        self.assertEqual(DetailTrackingPolicy(weak_match_zone_enabled=False).decision_zone(.75), "low")
        no_motion = DetailTrackingPolicy(motion_continuity_enabled=False).rank_profiles(
            np.asarray([1, 0], dtype=np.float32), None, [candidate],
            exclude_person_ids=set(), bbox_xyxy=(10, 10, 30, 60), frame_index=10,
            image_width=100, image_height=100,
            recent_person_positions={"person": (9, (20.0, 35.0))},
        )[0]
        self.assertEqual(no_motion.motion_bonus, 0)


class DetailPersistenceAndReportTests(unittest.TestCase):
    def test_d4_creates_after_initial_buffer_while_d1_keeps_low_match_pending(self) -> None:
        def run(delayed: bool) -> tuple[int, str]:
            with tempfile.TemporaryDirectory() as folder:
                paths = paths_for(Path(folder))
                detection = Detection(1, (10, 5, 80, 155), .95)
                capture = FakeCapture(1)
                capture.frames = [np.full((160, 100, 3), 120, dtype=np.uint8)]
                pipeline = PersonReIdPipeline(
                    PipelineConfig(
                        encoder_backend="colorhist", max_frames=0, draw_debug=False,
                        decision_policy="details_tracking_v2",
                        delayed_new_person_enabled=delayed,
                        min_crop_width=1, min_crop_height=1, crop_padding=0,
                        min_good_frames_before_reid=1, min_embedding_quality=0,
                        min_update_quality=0, reid_every_n_frames=1,
                    ),
                    paths=paths, tracker=SequenceTracker([[detection]]), encoder=RecordingEncoder(),
                )
                pipeline.profiles.add_or_update_person(
                    "person_000001", np.asarray([-1.0, 0.0], dtype=np.float32),
                    "seed", 0, 99, (0, 0, 10, 20), None, None,
                    {"quality_score": 1.0},
                )
                pipeline._open_capture = lambda _source: capture
                with patch("app.pipeline.orchestrator.cv2.VideoWriter", return_value=FakeWriter()), patch(
                    "app.pipeline.orchestrator.save_crop", return_value=Path(folder) / "snapshot.jpg"
                ):
                    result = pipeline.process("fixture.mp4")
                row = json.loads(result.predictions_path.read_text(encoding="utf-8").splitlines()[0])
                return result.created_persons, row["detections"][0]["state"]

        self.assertEqual(run(True), (0, "pending_new_person"))
        self.assertEqual(run(False), (1, "created_identity"))

    def test_pipeline_policy_persists_details_and_reports_decision_counters(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            detection = Detection(1, (10, 5, 80, 155), .95)
            capture = FakeCapture(2)
            capture.frames = [np.full((160, 100, 3), 120, dtype=np.uint8) for _ in range(2)]
            writer = FakeWriter()
            pipeline = PersonReIdPipeline(
                PipelineConfig(encoder_backend="colorhist", max_frames=0, draw_debug=False,
                               decision_policy="details_tracking_v2",
                               min_crop_width=1, min_crop_height=1, crop_padding=0,
                               min_good_frames_before_reid=1, min_embedding_quality=0,
                               min_update_quality=0, reid_every_n_frames=1),
                paths=paths, tracker=SequenceTracker([[detection], [detection]]),
                encoder=RecordingEncoder(),
            )
            pipeline._open_capture = lambda _source: capture
            with patch("app.pipeline.orchestrator.cv2.VideoWriter", return_value=writer), patch(
                "app.pipeline.orchestrator.save_crop", return_value=Path(folder) / "snapshot.jpg"
            ):
                result = pipeline.process("fixture.mp4")
            self.assertEqual(result.decision_policy, "details_tracking_v2")
            self.assertEqual(result.created_persons, 1)
            stored = pipeline.store.get_profile("person_000001")
            self.assertIsNotNone(stored.detail_vector)
            self.assertEqual(stored.detail_vector.size, DETAIL_VECTOR_LENGTH)
            events = pipeline.store.events_dataframe()
            self.assertIn("details_label", events.columns)

    def test_sqlite_profile_round_trip_preserves_detail_vector(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            store = SQLiteVectorStore(Path(folder) / "reid.sqlite3")
            service = ProfileService(store)
            detail = np.linspace(0, 1, DETAIL_VECTOR_LENGTH, dtype=np.float32)
            service.add_or_update_person(
                "person_000001", np.asarray([1, 0], dtype=np.float32), "fixture", 1, 1,
                (0, 0, 10, 20), None, None,
                {"quality_score": .8, "detail_vector": detail.tolist(), "detail_weight": .8},
            )
            reopened = SQLiteVectorStore(Path(folder) / "reid.sqlite3").get_profile("person_000001")
            np.testing.assert_allclose(reopened.detail_vector, detail)
            self.assertEqual(reopened.detail_weight_sum, .8)

    def test_main_frame_artifact_is_converted_to_single_person_report(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            predictions = root / "frames.jsonl"
            rows = [
                {"frame_index": 1, "detections": [{"track_id": 7, "person_id": "person_000001", "state": "created_identity", "decision_zone": "empty_db"}]},
                {"frame_index": 2, "detections": [{"track_id": 7, "person_id": "person_000001", "state": "known_track", "decision_zone": None}]},
            ]
            predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            events, frames = load_prediction_events(predictions)
            self.assertEqual(frames, 2)
            self.assertEqual(events[0]["event_type"], "new_person")
            report = create_single_person_report(
                predictions_path=predictions, output_dir=root / "report", video_name="one.mp4",
                expected_person_id="person_000001",
            )
            self.assertEqual(report.metrics["dominant_person_ratio"], 1.0)
            self.assertEqual(report.metrics["fragmentation_rating"], "gut")
            self.assertTrue(report.markdown_path.is_file())


if __name__ == "__main__":
    unittest.main()
