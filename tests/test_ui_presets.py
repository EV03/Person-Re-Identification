from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.test_evaluation_scope import paths_for


class UiPresetTests(unittest.TestCase):
    def test_reference_and_comparison_presets_render_with_correct_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            with patch("app.config.AppPaths", return_value=paths):
                app = AppTest.from_file("app/ui/streamlit_app.py", default_timeout=20).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(next(w for w in app.selectbox if w.label == "Encoder backend").value, "torchreid")
                for preset, encoder, quality in (
                    ("colorhist", "colorhist", .55),
                    ("no_quality_thresholds", "torchreid", 0.0),
                ):
                    next(w for w in app.selectbox if w.label == "ReID preset").select(preset).run()
                    self.assertEqual(len(app.exception), 0)
                    self.assertEqual(next(w for w in app.selectbox if w.label == "Encoder backend").value, encoder)
                    self.assertEqual(next(w for w in app.slider if w.label == "Min crop quality for ReID candidates").value, quality)
                labels = [w.label for w in app.checkbox] + [w.label for w in app.selectbox]
                self.assertFalse(any("motion" in label.lower() or "football" in label.lower() for label in labels))
                self.assertEqual(len(list(paths.mode_dir.glob("*.json"))), 0)


if __name__ == "__main__":
    unittest.main()
