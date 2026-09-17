from __future__ import annotations

import unittest

from app.pipeline.orchestrator import has_sufficient_new_person_evidence, intersection_over_smaller_box


class DetectionOverlapTests(unittest.TestCase):
    def test_nested_or_shifted_duplicate_boxes_overlap_strongly(self) -> None:
        full_person = (103, 70, 332, 837)
        shifted_body = (92, 212, 332, 947)

        self.assertGreater(intersection_over_smaller_box(full_person, shifted_body), 0.65)

    def test_separate_people_do_not_overlap(self) -> None:
        left_person = (10, 20, 100, 300)
        right_person = (130, 25, 220, 305)

        self.assertEqual(intersection_over_smaller_box(left_person, right_person), 0.0)

    def test_invalid_box_is_safe(self) -> None:
        self.assertEqual(intersection_over_smaller_box((10, 10, 10, 40), (0, 0, 20, 20)), 0.0)

    def test_short_false_positive_does_not_create_person(self) -> None:
        evidence = [(frame, 0.20) for frame in range(287, 297)]

        self.assertFalse(
            has_sufficient_new_person_evidence(
                evidence,
                max_score=0.58,
                min_events=6,
                min_span_frames=15,
                low_match_ratio=0.80,
            )
        )

    def test_persistent_low_match_can_create_person(self) -> None:
        evidence = [(1, 0.20), (4, 0.25), (7, 0.30), (10, 0.22), (13, 0.18), (16, 0.27)]

        self.assertTrue(
            has_sufficient_new_person_evidence(
                evidence,
                max_score=0.58,
                min_events=6,
                min_span_frames=15,
                low_match_ratio=0.80,
            )
        )


if __name__ == "__main__":
    unittest.main()
