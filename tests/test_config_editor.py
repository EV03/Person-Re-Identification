from __future__ import annotations

import unittest

from app.modes.default_mode import build_default_mode
from app.ui.config_editor import (
    build_run_config,
    changed_parameters,
    preset_from_config,
    runtime_parameters,
)


class ConfigEditorTests(unittest.TestCase):
    def test_unchanged_config_keeps_its_reference_name(self):
        mode = build_default_mode()
        parameters = runtime_parameters(mode.to_pipeline_config())
        self.assertFalse(changed_parameters(mode, parameters))
        self.assertEqual(build_run_config(mode, parameters).mode_name, mode.name)

    def test_run_and_saved_preset_share_the_same_runtime_parameters(self):
        mode = build_default_mode()
        parameters = runtime_parameters(mode.to_pipeline_config())
        parameters.update(min_embedding_quality=.7123, min_crop_width=12, crop_padding=.2)
        config = build_run_config(mode, parameters)
        self.assertEqual(set(changed_parameters(mode, parameters)), {"min_embedding_quality", "min_crop_width", "crop_padding"})
        self.assertIn("geändert", config.mode_name)
        saved = preset_from_config(config, mode_id="pilot", name="Pilot", description="test")
        self.assertEqual(runtime_parameters(saved.to_pipeline_config()), parameters)
        self.assertEqual(saved.to_pipeline_config().mode_name, "Pilot")
        self.assertTrue(saved.is_custom)
        self.assertEqual(mode.min_embedding_quality, .55)

    def test_incomplete_or_extra_parameters_are_rejected(self):
        mode = build_default_mode()
        parameters = runtime_parameters(mode.to_pipeline_config())
        for invalid in (
            {name: value for name, value in parameters.items() if name != "min_crop_width"},
            {**parameters, "mode_name": "Mislabelled B0"},
        ):
            with self.assertRaises(ValueError):
                build_run_config(mode, invalid)
