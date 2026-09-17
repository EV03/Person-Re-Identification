from __future__ import annotations

import unittest
from unittest.mock import patch

from app.config import PipelineConfig
from app.modes.base_mode import ModeConfig
from app.modes.mode_registry import list_modes


class ModeConfigTests(unittest.TestCase):
    def test_all_modes_accept_runtime_calibration_values(self) -> None:
        for mode in list_modes().values():
            with self.subTest(mode=mode.mode_id):
                config = mode.to_pipeline_config(
                    calibration_mode="new_person",
                    calibration_target_person_id="",
                    calibration_label="browser-run",
                )

                self.assertIsInstance(config, PipelineConfig)
                self.assertEqual(config.mode_name, mode.name)
                self.assertEqual(config.calibration_mode, "new_person")
                self.assertEqual(config.calibration_label, "browser-run")

    def test_stale_constructor_signature_does_not_break_runtime_fields(self) -> None:
        class LegacyPipelineConfig:
            __dataclass_fields__ = {"mode_id": object(), "mode_name": object()}

            def __init__(self) -> None:
                self.mode_id = "default"
                self.mode_name = "Default ReID MVP"

        mode = ModeConfig(mode_id="test", name="Test mode", description="")
        with patch("app.modes.base_mode.PipelineConfig", LegacyPipelineConfig):
            config = mode.to_pipeline_config(
                calibration_mode="off",
                calibration_target_person_id="",
                calibration_label="",
            )

        self.assertEqual(config.mode_id, "test")
        self.assertEqual(config.mode_name, "Test mode")
        self.assertEqual(config.calibration_mode, "off")


if __name__ == "__main__":
    unittest.main()
