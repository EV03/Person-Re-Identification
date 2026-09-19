from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.modes.base_mode import ModeConfig
from app.modes.mode_registry import builtin_modes, list_modes, save_custom_mode
from app.storage.models import PipelineResult
from app.ui.config_editor import RUNTIME_PARAMETER_FIELDS, runtime_parameters
from tests.test_evaluation_scope import paths_for


def keyed_widget(app, key):
    for kind in ("number_input", "text_input", "selectbox", "checkbox"):
        for widget in getattr(app, kind):
            if widget.key == key:
                return widget
    raise AssertionError(f"Missing widget: {key}")


EDITED_PARAMETERS = {
    "decision_policy": "details_tracking_v2",
    "yolo_model": "yolov8n.pt",
    "tracker": "pilot_tracker.yaml",
    "encoder_backend": "colorhist",
    "reid_model_name": "osnet_x0_5",
    "reid_checkpoint": "data/models/osnet_x1_0_msmt17.pth",
    "match_threshold": .731,
    "detection_confidence": .1234,
    "image_size": 736,
    "reid_every_n_frames": 7,
    "min_good_frames_before_reid": 4,
    "initial_candidate_every_n_frames": 2,
    "min_embedding_quality": .4217,
    "min_initial_blur_score": .4012,
    "min_border_blur_score": .4567,
    "min_update_quality": .8765,
    "min_update_similarity": .8234,
    "max_frames": 0,
    "min_crop_height": 51,
    "min_crop_width": 17,
    "crop_padding": .12,
    "max_person_overlap_ratio": .2345,
    "overlap_cooldown_frames": 17,
    "track_state_ttl_frames": 23,
    "device": "cpu",
    "draw_debug": False,
    "live_preview_every_n_frames": 13,
    "detail_reranking_enabled": False,
    "weak_match_zone_enabled": False,
    "delayed_new_person_enabled": False,
    "overlap_protection_enabled": False,
    "motion_continuity_enabled": False,
    "detail_weight": .12,
    "detail_min_confidence": .52,
    "strong_match_threshold": .84,
    "weak_match_threshold": .69,
    "new_person_max_score": .57,
    "new_person_min_evidence_events": 5,
    "new_person_min_evidence_span_frames": 12,
    "new_person_evidence_window_frames": 28,
    "new_person_low_match_ratio": .75,
    "new_person_overlap_threshold": .60,
    "motion_identity_bonus": .03,
    "motion_identity_max_frame_gap": 12,
    "motion_identity_max_distance_fraction": .12,
}


def edit_every_parameter(app):
    for field, value in EDITED_PARAMETERS.items():
        widget = keyed_widget(app, f"pipeline_{field}")
        if field in {"yolo_model", "encoder_backend", "reid_model_name", "reid_checkpoint", "decision_policy"}:
            widget.select(value)
        else:
            widget.set_value(value)
    app.run()
    assert not app.exception


class UiPresetTests(unittest.TestCase):
    def test_ui_exposes_separate_multi_and_single_person_test_modules(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch("app.config.AppPaths", return_value=paths_for(Path(folder))):
            app = AppTest.from_file("app/ui/streamlit_app.py").run()
            module = next(widget for widget in app.selectbox if widget.label == "Test module")
            self.assertEqual(module.options, ["Mehrpersonen- und Trackingtest", "Einzelpersonen- und Detailtest"])
            self.assertFalse(any(widget.label == "Details_Tracking-Policy auch für 2+ Personen verwenden"
                                 for widget in app.checkbox))
            self.assertEqual(keyed_widget(app, "pipeline_decision_policy").value, "main_single_threshold")
            module.select("Einzelpersonen- und Detailtest").run()
            self.assertEqual(keyed_widget(app, "pipeline_decision_policy").value, "main_single_threshold")
            preset = next(widget for widget in app.selectbox if widget.label == "ReID preset")
            self.assertTrue(any("(details_tracking)" in option for option in preset.options))
            preset.select("details_tracking").run()
            self.assertEqual(keyed_widget(app, "pipeline_decision_policy").value, "details_tracking_v2")
            self.assertFalse(app.exception)
            self.assertEqual(keyed_widget(app, "pipeline_decision_policy").value, "details_tracking_v2")
            self.assertTrue(any(widget.label == "Expected person ID (optional)" for widget in app.text_input))
            self.assertTrue(any(widget.label == "Test condition" for widget in app.text_input))

    def test_nonisolated_runs_switch_encoder_database_and_return_to_original(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            with patch("app.config.AppPaths", return_value=paths), patch(
                "app.utils.camera_utils.scan_local_cameras", return_value=[]
            ), patch("app.pipeline.orchestrator.PersonReIdPipeline") as constructor:
                constructor.return_value.process.return_value = PipelineResult(
                    output_video_path=None, processed_frames=1, created_persons=0, matched_events=0)
                app = AppTest.from_file("app/ui/streamlit_app.py").run()
                next(w for w in app.checkbox if w.label == "Isolierter Lauf (neue Datenbank)").uncheck().run()
                next(w for w in app.radio if w.label == "Input type").set_value("Local webcam").run()
                next(w for w in app.checkbox if w.label == "Use manual camera index").check().run()
                databases = []
                for backend in ("torchreid", "colorhist", "torchreid"):
                    keyed_widget(app, "pipeline_encoder_backend").select(backend).run()
                    next(w for w in app.button if w.label.startswith("Run ")).click().run()
                    self.assertFalse(app.exception)
                    databases.append(constructor.call_args.kwargs["paths"].db_path)
                self.assertNotEqual(databases[0], databases[1])
                self.assertEqual(databases[0], databases[2])

    def test_reference_and_comparison_presets_render_with_correct_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            with patch("app.config.AppPaths", return_value=paths):
                app = AppTest.from_file("app/ui/streamlit_app.py", default_timeout=20).run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(next(w for w in app.selectbox if w.label == "Encoder backend").value, "torchreid")
                self.assertTrue(any(w.label == "YOLO model" for w in app.selectbox))
                self.assertTrue(any(w.label == "ReID model" for w in app.selectbox))
                self.assertTrue(any(w.label == "ReID checkpoint" for w in app.selectbox))
                for preset, encoder, quality, update_similarity in (
                    ("colorhist", "colorhist", .55, .82),
                    ("no_quality_thresholds", "torchreid", 0.0, .82),
                    ("no_update_similarity", "torchreid", .55, -1.0),
                ):
                    next(w for w in app.selectbox if w.label == "ReID preset").select(preset).run()
                    self.assertEqual(len(app.exception), 0)
                    self.assertEqual(next(w for w in app.selectbox if w.label == "Encoder backend").value, encoder)
                    self.assertEqual(keyed_widget(app, "pipeline_min_embedding_quality").value, quality)
                    self.assertEqual(keyed_widget(app, "pipeline_min_update_similarity").value, update_similarity)
                next(w for w in app.selectbox if w.label == "ReID preset").select("details_tracking").run()
                self.assertEqual(keyed_widget(app, "pipeline_decision_policy").value, "details_tracking_v2")
                labels = [w.label for w in app.checkbox] + [w.label for w in app.selectbox]
                self.assertFalse(any("football" in label.lower() for label in labels))
                self.assertEqual(len(list(paths.mode_dir.glob("*.json"))), 0)

    def test_editor_has_one_control_for_every_runtime_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch("app.config.AppPaths", return_value=paths_for(Path(folder))):
            app = AppTest.from_file("app/ui/streamlit_app.py").run()
            keys = [w.key for kind in ("number_input", "text_input", "selectbox", "checkbox") for w in getattr(app, kind)]
            pipeline_keys = [key for key in keys if key and key.startswith("pipeline_")]
            self.assertCountEqual(pipeline_keys, [f"pipeline_{field}" for field in RUNTIME_PARAMETER_FIELDS])
            self.assertFalse(any(w.label == "Base mode" for w in app.selectbox))
            self.assertEqual(len(app.get("form")), 1)

    def test_edits_survive_reruns_and_can_be_discarded_or_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch("app.config.AppPaths", return_value=paths_for(Path(folder))):
            app = AppTest.from_file("app/ui/streamlit_app.py").run()
            edit_every_parameter(app)
            app.run()
            for field, value in EDITED_PARAMETERS.items():
                self.assertEqual(keyed_widget(app, f"pipeline_{field}").value, value)
            self.assertTrue(any("geändert" in w.value for w in app.warning))
            self.assertTrue(any("geändert" in w.label for w in app.button if w.label.startswith("Run ")))
            next(w for w in app.button if w.label == "Änderungen verwerfen / Preset neu laden").click().run()
            baseline = runtime_parameters(builtin_modes()["default"].to_pipeline_config())
            for field, value in baseline.items():
                self.assertEqual(keyed_widget(app, f"pipeline_{field}").value, value)
            self.assertFalse(app.warning)
            edit_every_parameter(app)
            keyed_widget(app, "selected_preset_id").select("no_quality_thresholds").run()
            expected = runtime_parameters(builtin_modes()["no_quality_thresholds"].to_pipeline_config())
            for field, value in expected.items():
                self.assertEqual(keyed_widget(app, f"pipeline_{field}").value, value)

    def test_saving_captures_every_edit_and_loads_the_new_preset(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            with patch("app.config.AppPaths", return_value=paths):
                app = AppTest.from_file("app/ui/streamlit_app.py").run()
                edit_every_parameter(app)
                next(w for w in app.text_input if w.label == "Mode id").set_value("pilot")
                next(w for w in app.text_input if w.label == "Mode name").set_value("Pilot exact")
                next(w for w in app.text_area if w.label == "Description").set_value("All parameters edited")
                next(w for w in app.button if w.label == "Aktuelle Einstellungen speichern").click().run()
                self.assertFalse(app.exception)
                saved = list_modes(paths)["pilot"]
                self.assertEqual(runtime_parameters(saved.to_pipeline_config()), EDITED_PARAMETERS)
                self.assertEqual(saved.name, "Pilot exact")
                self.assertEqual(saved.description, "All parameters edited")
                self.assertEqual(keyed_widget(app, "selected_preset_id").value, "pilot")
                self.assertFalse(app.warning)
                keyed_widget(app, "selected_preset_id").select("default").run()
                keyed_widget(app, "selected_preset_id").select("pilot").run()
                for field, value in EDITED_PARAMETERS.items():
                    self.assertEqual(keyed_widget(app, f"pipeline_{field}").value, value)
                saved_contents = paths.mode_config_path.read_bytes()
                keyed_widget(app, "pipeline_match_threshold").set_value(.3).run()
                next(w for w in app.text_input if w.label == "Mode id").set_value("pilot")
                next(w for w in app.button if w.label == "Aktuelle Einstellungen speichern").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(app.error)
                self.assertEqual(paths.mode_config_path.read_bytes(), saved_contents)
                self.assertEqual(keyed_widget(app, "pipeline_match_threshold").value, .3)

    def test_saving_cannot_overwrite_a_reference_preset(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            with patch("app.config.AppPaths", return_value=paths):
                app = AppTest.from_file("app/ui/streamlit_app.py").run()
                edit_every_parameter(app)
                next(w for w in app.text_input if w.label == "Mode id").set_value("default")
                next(w for w in app.button if w.label == "Aktuelle Einstellungen speichern").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(app.error)
                self.assertFalse(paths.mode_config_path.exists())
                self.assertEqual(list_modes(paths)["default"].match_threshold, .75)
                self.assertEqual(keyed_widget(app, "pipeline_match_threshold").value, .731)

    def test_custom_preset_can_be_updated_with_current_editor_values(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            save_custom_mode(ModeConfig("pilot", "Pilot alt", "Alte Beschreibung"), paths)
            with patch("app.config.AppPaths", return_value=paths):
                app = AppTest.from_file("app/ui/streamlit_app.py").run()
                self.assertFalse(any(
                    button.label == "Ausgewählte Konfiguration aktualisieren"
                    for button in app.button
                ))
                keyed_widget(app, "selected_preset_id").select("pilot").run()
                keyed_widget(app, "pipeline_min_good_frames_before_reid").set_value(6)
                keyed_widget(app, "pipeline_initial_candidate_every_n_frames").set_value(4)
                keyed_widget(app, "pipeline_reid_every_n_frames").set_value(12)
                app.run()
                next(w for w in app.text_input if w.label == "Gespeicherter Name").set_value("Pilot aktualisiert")
                next(w for w in app.text_area if w.label == "Gespeicherte Beschreibung").set_value("Neue Werte")
                next(
                    w for w in app.button
                    if w.label == "Ausgewählte Konfiguration aktualisieren"
                ).click().run()

                self.assertFalse(app.exception)
                saved = list_modes(paths)["pilot"]
                self.assertEqual(saved.name, "Pilot aktualisiert")
                self.assertEqual(saved.description, "Neue Werte")
                self.assertEqual(saved.min_good_frames_before_reid, 6)
                self.assertEqual(saved.initial_candidate_every_n_frames, 4)
                self.assertEqual(saved.reid_every_n_frames, 12)
                self.assertEqual(keyed_widget(app, "selected_preset_id").value, "pilot")
                self.assertFalse(app.warning)

    def test_threshold_endpoints_and_precise_values_are_editable(self) -> None:
        with tempfile.TemporaryDirectory() as folder, patch("app.config.AppPaths", return_value=paths_for(Path(folder))):
            app = AppTest.from_file("app/ui/streamlit_app.py").run()
            for values in (
                {"match_threshold": -1.0, "detection_confidence": .0001, "min_embedding_quality": 0.0, "min_initial_blur_score": 0.0, "min_border_blur_score": 0.0, "min_update_quality": 0.0, "min_crop_width": 0, "min_crop_height": 0},
                {"match_threshold": 1.0, "detection_confidence": 1.0, "min_embedding_quality": 1.0, "min_initial_blur_score": 1.0, "min_border_blur_score": 1.0, "min_update_quality": 1.0},
                {"match_threshold": .8123, "detection_confidence": .2345, "min_embedding_quality": .5678, "min_initial_blur_score": .3456, "min_border_blur_score": .4567, "min_update_quality": .6789},
            ):
                for field, value in values.items():
                    keyed_widget(app, f"pipeline_{field}").set_value(value)
                app.run()
                self.assertFalse(app.exception)
                for field, value in values.items():
                    self.assertEqual(keyed_widget(app, f"pipeline_{field}").value, value)

    def test_start_uses_exactly_the_current_editor_values(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = paths_for(Path(folder))
            with patch("app.config.AppPaths", return_value=paths), patch(
                "app.utils.camera_utils.scan_local_cameras", return_value=[]
            ), patch("app.pipeline.orchestrator.PersonReIdPipeline") as constructor:
                constructor.return_value.process.return_value = PipelineResult(
                    output_video_path=None, processed_frames=3, created_persons=1, matched_events=0
                )
                app = AppTest.from_file("app/ui/streamlit_app.py").run()
                edit_every_parameter(app)
                next(w for w in app.radio if w.label == "Input type").set_value("Local webcam").run()
                next(w for w in app.checkbox if w.label == "Use manual camera index").check().run()
                duration = next(w for w in app.number_input if w.label == "Live-Aufnahmedauer (Sekunden)")
                self.assertEqual(duration.value, 30)
                duration.set_value(45).run()
                next(w for w in app.button if w.label.startswith("Run ")).click().run()
                self.assertFalse(app.exception)
                constructor.assert_called_once()
                config = constructor.call_args.kwargs["config"]
                self.assertEqual(runtime_parameters(config), EDITED_PARAMETERS)
                self.assertEqual(config.mode_name, "B0 - OSNet (geändert)")
                constructor.return_value.process.assert_called_once()
                self.assertEqual(constructor.return_value.process.call_args.kwargs["max_duration_seconds"], 45.0)


if __name__ == "__main__":
    unittest.main()
