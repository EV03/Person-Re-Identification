"""Streamlit entry point for configuring, running and inspecting analyses.

Streamlit executes this module from top to bottom on every interaction.  Values
that must survive a rerun therefore belong in ``st.session_state`` or a cache;
pipeline work should only start inside an explicit user-action branch.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

# Allow running this file directly via: streamlit run app/ui/streamlit_app.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import streamlit as st

from app.config import AppPaths, PipelineConfig
from app.modes.base_mode import ModeConfig
from app.modes.mode_registry import list_modes, normalize_mode_id, save_custom_mode
from app.pipeline.orchestrator import PersonReIdPipeline
from app.storage.vector_store import SQLiteVectorStore
from app.evaluation.runner import create_unit_paths
from app.ui.config_editor import (
    RUNTIME_PARAMETER_FIELDS,
    build_run_config,
    changed_parameters,
    preset_from_config,
    runtime_parameters,
)
from app.utils.camera_utils import (
    CameraSource,
    camera_backends,
    read_single_preview_frame,
    scan_local_cameras,
)
from app.utils.upload_utils import persist_uploaded_video


@st.cache_data(ttl=20, show_spinner=False)
def cached_scan_local_cameras(max_index: int, backend_name: str) -> list[dict[str, object]]:
    cameras = scan_local_cameras(max_index=max_index, backend_name=backend_name)
    return [
        {
            "index": cam.index,
            "backend_name": cam.backend_name,
            "backend_id": cam.backend_id,
            "width": cam.width,
            "height": cam.height,
            "fps": cam.fps,
            "label": cam.label,
            "key": cam.key,
        }
        for cam in cameras
    ]


def bgr_to_rgb(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def load_editor(mode: ModeConfig) -> None:
    """Load before rendering widgets; explicit keys keep edits across reruns."""
    for field, value in runtime_parameters(mode.to_pipeline_config()).items():
        st.session_state[f"pipeline_{field}"] = value
    st.session_state["editor_preset_id"] = mode.mode_id


def reset_editor() -> None:
    # Button callbacks run before the next script execution and widget creation.
    st.session_state["editor_preset_id"] = None


def render_pipeline_editor() -> dict[str, object]:
    st.subheader("Modelle und Tracking")
    st.text_input("YOLO model", key="pipeline_yolo_model")
    st.text_input(
        "Tracker configuration", key="pipeline_tracker",
        help="bytetrack.yaml, botsort.yaml oder eine eigene YAML-Datei. Interne Tracker-Schwellen werden in dieser Datei eingestellt.",
    )
    st.selectbox("Encoder backend", ["colorhist", "torchreid"], key="pipeline_encoder_backend")
    st.text_input("ReID model", key="pipeline_reid_model_name")
    st.text_input("ReID checkpoint", key="pipeline_reid_checkpoint",
                  help="Explizite ReID-Gewichte für OSNet. Relative Pfade beziehen sich auf das Projekt; Farbhistogramme ignorieren dieses Feld.")
    st.text_input("Device", key="pipeline_device", help="auto, cpu, cuda oder cuda:0")

    st.subheader("Schwellenwerte")
    for field, label, lower, upper, help_text in (
        ("match_threshold", "Match threshold", -1.0, 1.0, "Cosine Similarity: Ein Profil wird ab diesem Wert akzeptiert."),
        # Ultralytics' track() replaces an exact zero with its 0.1 default.
        ("detection_confidence", "Detection confidence", 0.0001, 1.0, "Konfidenzgrenze für YOLO. Muss positiv sein: Ultralytics würde exakt 0 intern durch 0,1 ersetzen."),
        ("min_embedding_quality", "Min crop quality for ReID candidates", 0.0, 1.0, "0 deaktiviert diese Qualitätsschwelle; Mindestgrößen und Qualitätsgewichtung bleiben erhalten."),
        ("min_update_quality", "Min crop quality for person embedding updates", 0.0, 1.0, "Updates müssen zusätzlich die Kandidatenschwelle erfüllen."),
    ):
        st.number_input(label, min_value=lower, max_value=upper, step=0.01,
                        format="%.4f", key=f"pipeline_{field}", help=help_text)

    st.subheader("Ausschnitte und zeitliche Parameter")
    for field, label, minimum, help_text in (
        ("min_crop_width", "Min crop width (px)", 0, "0 deaktiviert die Mindestbreite; leere Crops bleiben ungültig."),
        ("min_crop_height", "Min crop height (px)", 0, "0 deaktiviert die Mindesthöhe; leere Crops bleiben ungültig."),
        ("min_good_frames_before_reid", "Min good frames before first ReID match", 1, "Anzahl akzeptierter Beobachtungen vor der ersten Identitätsentscheidung."),
        ("reid_every_n_frames", "ReID every N frames", 1, "Update-Intervall bekannter Tracks; unbekannte Tracks sammeln Kandidaten in jedem Frame."),
        ("image_size", "Image size", 32, "YOLO-Eingangsgröße; Vielfache von 32 verwenden."),
        ("max_frames", "Max frames (0 = vollständiges Video)", 0, "Für vollständige Evaluationsclips 0 setzen."),
    ):
        st.number_input(label, min_value=minimum, step=1, key=f"pipeline_{field}", help=help_text)
    st.number_input("Crop padding", min_value=0.0, step=0.01, format="%.4f",
                    key="pipeline_crop_padding", help="Zusätzlicher Rand relativ zur Boxgröße, z. B. 0,05 = 5 % pro Seite.")

    st.subheader("Ausgabeparameter")
    st.checkbox("Annotate output video", key="pipeline_draw_debug")
    st.number_input("Preview every N frames", min_value=1, step=1,
                    key="pipeline_live_preview_every_n_frames")
    return {field: st.session_state[f"pipeline_{field}"] for field in RUNTIME_PARAMETER_FIELDS}


def render_save_mode_form(paths: AppPaths, config: PipelineConfig) -> None:
    with st.expander("Aktuelle Einstellungen als neue Versuchskonfiguration speichern"):
        st.caption("Speichert exakt alle oben eingestellten Pipeline-Parameter. B0/A1/A2 und vorhandene Presets werden nicht überschrieben.")
        with st.form("save_current_configuration"):
            custom_name = st.text_input("Mode name", value="Mein ReID-Pilot")
            custom_mode_id_raw = st.text_input("Mode id", value="mein_reid_pilot")
            custom_description = st.text_area("Description", height=80)
            submitted = st.form_submit_button("Aktuelle Einstellungen speichern")
        if submitted:
            try:
                mode_id = normalize_mode_id(custom_mode_id_raw)
                mode = preset_from_config(config, mode_id=mode_id,
                                          name=custom_name.strip() or mode_id,
                                          description=custom_description.strip())
                saved = save_custom_mode(mode, paths=paths, overwrite=False)
            except (OSError, TypeError, ValueError) as exc:
                st.error(str(exc))
            else:
                # Apply selection at the start of the rerun, before its widget exists.
                st.session_state["pending_preset_id"] = saved.mode_id
                st.session_state["preset_saved_notice"] = f"Gespeichert und geladen: {saved.name} ({saved.mode_id})"
                st.rerun()


st.set_page_config(page_title="Local Person ReID MVP", layout="wide")

paths = AppPaths()
paths.ensure()

st.title("Local Person Re-Identification MVP")
st.caption("Forschungsprototyp: YOLO, Tracking, qualitätsgefilterte ReID und lokale SQLite-Speicherung")
st.caption("B0/A1/A2: OSNet mit dokumentierten ReID-Gewichten, vollständiger Frame-Export und Laufmanifest. Parameter vor den Testclips einfrieren.")

modes = list_modes(paths)
pending_preset_id = st.session_state.pop("pending_preset_id", None)
if pending_preset_id in modes:
    st.session_state["selected_preset_id"] = pending_preset_id

with st.sidebar:
    st.header("Versuchskonfiguration")
    selected_mode_id = st.selectbox(
        "ReID preset",
        options=list(modes.keys()),
        format_func=lambda mode_id: f"{modes[mode_id].name} ({mode_id})",
        key="selected_preset_id",
    )
    selected_mode = modes[selected_mode_id]
    st.caption(selected_mode.description)

    if st.session_state.get("editor_preset_id") != selected_mode_id:
        load_editor(selected_mode)
    notice = st.session_state.pop("preset_saved_notice", None)
    if notice:
        st.success(notice)
    st.caption("Preset laden → Parameter bearbeiten → starten oder als neues Preset speichern. Nur die aktuellen Werte gelten beim Start.")
    st.button("Änderungen verwerfen / Preset neu laden", on_click=reset_editor)

    st.divider()
    st.header("Pipeline-Parameter")
    parameters = render_pipeline_editor()
    config = build_run_config(selected_mode, parameters)
    changes = changed_parameters(selected_mode, parameters)
    if changes:
        st.warning(f"{selected_mode.name}: geändert ({len(changes)} Parameter). Noch nicht als neues Preset gespeichert.")
        with st.expander("Änderungen gegenüber dem geladenen Preset"):
            st.json(changes)
    else:
        st.success(f"{selected_mode.name}: unverändert")
    if config.min_update_quality < config.min_embedding_quality:
        st.info("Die Update-Schwelle liegt unter der Kandidatenschwelle. Effektiv müssen Updates beide erfüllen; die höhere Schwelle gilt.")
    st.caption("Schwellen auf Pilotdaten einstellen und vor der Evaluation einfrieren. Die Qualitätsheuristik selbst bleibt unverändert.")
    render_save_mode_form(paths, config)
    with st.expander("Tatsächlich verwendete Pipeline-Konfiguration"):
        st.json(asdict(config))

    st.divider()
    st.header("Quelle und Anzeige (nicht Teil des Presets)")
    input_type = st.radio("Input type", ["Video upload", "Local webcam"], index=0)
    isolated_run = st.checkbox("Isolierter Lauf (neue Datenbank)", value=True,
                               help="Standard für unabhängige Versuche. Deaktivieren teilt den bisherigen interaktiven Profilbestand; für UC-12 den CLI-Versuchsstarter mit beiden Videos verwenden.")
    show_live_preview = st.checkbox("Show live annotated preview", value=True)
    preview_width = st.slider("Preview width", min_value=480, max_value=1400, value=960, step=40)

    st.divider()
    st.caption("Datenschutz-Hinweis: Das MVP nutzt synthetische IDs und sollte nur mit berechtigtem/consented Material getestet werden.")

source: str | int | CameraSource | None = None
uploaded_file = None
selected_camera_source: CameraSource | None = None

if input_type == "Video upload":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        try:
            input_path = persist_uploaded_video(paths.input_dir, uploaded_file.name, uploaded_file.getbuffer())
        except (OSError, ValueError) as exc:
            st.error(f"Could not store uploaded video: {exc}")
        else:
            source = str(input_path)
            st.subheader("Input video")
            st.video(str(input_path))
else:
    st.subheader("Local webcam")
    st.info(
        "Die Webcam wird auf dem Rechner geöffnet, auf dem Streamlit läuft. "
        "Das ist eine lokale OpenCV-Auswahl, nicht die Browser-MediaDevices-Auswahl."
    )

    backend_options = [name for name, _ in camera_backends()]
    default_backend_index = 0 if "dshow" in backend_options else len(backend_options) - 1

    cam_col_1, cam_col_2, cam_col_3 = st.columns([2, 2, 1])
    with cam_col_1:
        camera_backend = st.selectbox("Camera backend", backend_options, index=default_backend_index)
    with cam_col_2:
        max_camera_index = st.number_input("Scan camera indexes from 0 to", min_value=0, max_value=20, value=5, step=1)
    with cam_col_3:
        st.write("")
        st.write("")
        if st.button("Refresh", width="stretch"):
            cached_scan_local_cameras.clear()

    use_manual_camera_index = st.checkbox("Use manual camera index", value=False)

    if use_manual_camera_index:
        manual_col_1, manual_col_2 = st.columns([1, 1])
        with manual_col_1:
            webcam_index = st.number_input("Manual local webcam index", min_value=0, max_value=20, value=0, step=1)
        with manual_col_2:
            manual_backend = st.selectbox("Manual backend", backend_options, index=default_backend_index)
        selected_camera_source = CameraSource(index=int(webcam_index), backend_name=str(manual_backend))
    else:
        cameras = cached_scan_local_cameras(int(max_camera_index), str(camera_backend))
        if cameras:
            selected_camera_pos = st.selectbox(
                "Camera source",
                options=list(range(len(cameras))),
                format_func=lambda pos: str(cameras[int(pos)]["label"]),
            )
            selected_camera = cameras[int(selected_camera_pos)]
            selected_camera_source = CameraSource(
                index=int(selected_camera["index"]),
                backend_name=str(selected_camera["backend_name"]),
            )
        else:
            st.warning("No local OpenCV camera was found. Use manual index, another backend, or test a video upload first.")
            selected_camera_source = None

    if selected_camera_source is not None:
        source = selected_camera_source
        st.caption(f"Selected source: {selected_camera_source.label()}")

        preview_col_1, preview_col_2 = st.columns([1, 3])
        with preview_col_1:
            show_raw_preview = st.button("Test selected camera", width="stretch")
        with preview_col_2:
            st.caption("Dieser Test liest ein einzelnes Rohbild. Das Live-Tracking startet erst über 'Run re-identification'.")

        if show_raw_preview:
            ok, preview_frame = read_single_preview_frame(selected_camera_source)
            if ok and preview_frame is not None:
                st.image(bgr_to_rgb(preview_frame), caption="Selected camera raw preview", width=preview_width)
            else:
                st.error("Could not read a frame from the selected camera source.")

col_run, col_db = st.columns([1, 1])

with col_run:
    run_clicked = st.button(
        f"Run {config.mode_name}",
        type="primary",
        width="stretch",
        disabled=source is None,
    )

with col_db:
    displayed_paths = st.session_state.get("last_run_paths", paths)
    store = SQLiteVectorStore(displayed_paths.db_path)
    st.metric("Synthetic persons in displayed database", store.count_persons())

live_preview_placeholder = st.empty()
status_placeholder = st.empty()

if run_clicked and source is not None:
    progress = st.progress(0)
    status = status_placeholder

    def update_progress(current: int, total: int | None, message: str) -> None:
        # Streamlit can briefly lose the browser/WebSocket connection during long video runs.
        # Progress updates should not abort the actual pipeline processing.
        try:
            if total and total > 0:
                progress.progress(min(current / total, 1.0))
            status.write(message)
        except Exception:
            return

    def update_live_preview(frame_index: int, frame_bgr) -> None:
        if not show_live_preview:
            return
        # Sending too many preview frames can overload Streamlit on Windows and close the connection.
        # Ignore UI-only preview errors so the processing run can finish.
        try:
            live_preview_placeholder.image(
                bgr_to_rgb(frame_bgr),
                caption=f"Live annotated tracking preview - frame {frame_index}",
                width=int(preview_width),
            )
        except Exception:
            return

    pipeline = None
    try:
        run_paths = create_unit_paths(base_paths=paths, mode_id=config.mode_id) if isolated_run else paths
        st.session_state["last_run_paths"] = run_paths
        pipeline = PersonReIdPipeline(config=config, paths=run_paths)
        result = pipeline.process(source, progress_callback=update_progress, frame_callback=update_live_preview)
    except Exception as exc:
        st.error(str(exc))
        failure_manifest = getattr(pipeline, "last_manifest_path", None)
        if failure_manifest:
            st.caption(f"Fehlgeschlagener Lauf dokumentiert: {failure_manifest}")
        st.stop()

    st.success(f"Finished mode {result.mode_name}. Processed {result.processed_frames} frames.")
    if result.manifest_path:
        st.caption(f"Laufmanifest: {result.manifest_path}")
        st.caption(f"Vollständige Frame-Vorhersagen: {result.predictions_path}")

    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)

    left, right = st.columns([2, 1])
    with left:
        if result.output_video_path and result.output_video_path.exists():
            st.subheader("Annotated output video")
            st.video(str(result.output_video_path))

    with right:
        st.subheader("Run summary")
        st.metric("Mode", result.mode_id)
        st.metric("Created persons in this run", result.created_persons)
        st.metric("Matched events in this run", result.matched_events)
        st.metric("Total known persons", len(result.persons))

st.divider()

displayed_paths = st.session_state.get("last_run_paths", paths)
store = SQLiteVectorStore(displayed_paths.db_path)
st.caption(f"Angezeigte Datenbank: {displayed_paths.db_path}")
persons_df = store.persons_dataframe()
events_df = store.events_dataframe(limit=200)
runs_df = store.analysis_runs_dataframe(limit=100)

st.subheader("Analysis runs")
if runs_df.empty:
    st.info("No analysis runs stored yet.")
else:
    st.dataframe(runs_df, width="stretch")

st.subheader("Stored synthetic persons")
if persons_df.empty:
    st.info("No persons stored yet.")
else:
    st.dataframe(persons_df, width="stretch")

st.subheader("Recent events")
if events_df.empty:
    st.info("No events stored yet.")
else:
    st.dataframe(events_df, width="stretch")
