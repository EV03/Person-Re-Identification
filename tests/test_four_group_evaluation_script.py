from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile

from app.config import PipelineConfig
from scripts.run_four_group_evaluation import (
    DEFAULT_GROUPS,
    build_variants,
    extract_sources,
    parse_group_overrides,
    validate_sources,
)


class FourGroupEvaluationScriptTests(unittest.TestCase):
    def test_variants_are_derived_from_one_baseline(self) -> None:
        baseline = PipelineConfig(
            mode_id="calibrated",
            mode_name="Calibrated",
            match_threshold=.75,
            min_update_similarity=.75,
            reid_every_n_frames=5,
            overlap_cooldown_frames=11,
        )
        variants = build_variants(baseline, "cpu")

        self.assertEqual(tuple(variants), ("B0", "A2", "A3"))
        self.assertEqual(variants["B0"].match_threshold, .75)
        self.assertEqual(variants["B0"].min_update_similarity, .75)
        self.assertEqual(variants["B0"].reid_every_n_frames, 5)
        self.assertEqual(variants["B0"].overlap_cooldown_frames, 11)
        self.assertEqual(variants["B0"].max_frames, 0)
        self.assertEqual(variants["B0"].device, "cpu")

        b0 = asdict(variants["B0"])
        a2 = asdict(variants["A2"])
        a3 = asdict(variants["A3"])
        identity = {"mode_id", "mode_name"}
        a2_changes = {key for key in b0 if b0[key] != a2[key]} - identity
        a3_changes = {key for key in b0 if b0[key] != a3[key]} - identity
        self.assertEqual(
            a2_changes,
            {
                "min_embedding_quality",
                "min_initial_blur_score",
                "min_border_blur_score",
                "min_initial_aspect_ratio_score",
                "min_update_quality",
            },
        )
        self.assertEqual(a3_changes, {"min_update_similarity"})

    def test_zip_validation_and_extraction_use_exact_group_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "videos.zip"
            with ZipFile(archive, "w") as output:
                for group in DEFAULT_GROUPS.values():
                    output.writestr(f"nested/{group.filename}", group.group.encode("ascii"))

            validate_sources(archive, DEFAULT_GROUPS)
            extracted = extract_sources(archive, DEFAULT_GROUPS, root / "input")

            self.assertEqual(set(extracted), set(DEFAULT_GROUPS))
            for group, path in extracted.items():
                self.assertTrue(path.name.startswith(f"{group}_"))
                self.assertEqual(path.read_bytes(), group.encode("ascii"))

    def test_group_override_changes_only_the_requested_filename(self) -> None:
        groups = parse_group_overrides(["G2=return.mp4"])
        self.assertEqual(groups["G2"].filename, "return.mp4")
        self.assertEqual(groups["G1"], DEFAULT_GROUPS["G1"])


if __name__ == "__main__":
    unittest.main()
