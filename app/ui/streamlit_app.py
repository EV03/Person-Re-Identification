from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly via: streamlit run app/ui/streamlit_app.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pandas as pd
import streamlit as st

from app.config import AppPaths
from app.modes.base_mode import ModeConfig
from app.modes.mode_registry import list_modes, normalize_mode_id, save_custom_mode
from app.pipeline.orchestrator import PersonReIdPipeline
from app.storage.vector_store import SQLiteVectorStore
from app.utils.camera_utils import (
    CameraSource,
    camera_backends,
    read_single_preview_frame,
    scan_local_cameras,
)
from app.utils.id_utils import ensure_unique_path


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


def option_index(options: list[str], value: str, fallback: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return fallback


def render_create_mode_form(paths: AppPaths, modes: dict[str, ModeConfig]) -> None:
    with st.expander("Create custom mode preset"):
        st.caption(
            "Ein Custom Mode speichert nur Konfiguration und Feature-Flags. "
            "Die eigentliche Pipeline bleibt zunächst stabil und kann später modular erweitert werden."
        )
        with st.form("create_custom_mode_form"):
            base_mode_id = st.selectbox(
                "Base mode",
                options=list(modes.keys()),
                format_func=lambda mode_id: f"{modes[mode_id].name} ({mode_id})",
            )
            base_mode = modes[base_mode_id]
            custom_name = st.text_input("Mode name", value="My Football Variant")
            custom_mode_id_raw = st.text_input("Mode id", value="my_football_variant")
            custom_description = st.text_area(
                "Description",
                value="Custom preset for a specific video analysis setup.",
                height=80,
            )
            custom_pipeline_type = st.selectbox(
                "Pipeline type",
                options=["person_reid", "football_analysis"],
                index=option_index(["person_reid", "football_analysis"], base_mode.pipeline_type),
            )

            col_a, col_b = st.columns(2)
            with col_a:
                custom_yolo_model = st.text_input("YOLO model default", value=base_mode.yolo_model)
                custom_tracker = st.selectbox(
                    "Tracker default",
                    options=["bytetrack.yaml", "botsort.yaml"],
                    index=option_index(["bytetrack.yaml", "botsort.yaml"], base_mode.tracker),
                )
                custom_encoder = st.selectbox(
                    "Encoder default",
                    options=["colorhist", "torchreid"],
                    index=option_index(["colorhist", "torchreid"], base_mode.encoder_backend),
                )
            with col_b:
                custom_threshold = st.slider(
                    "Match threshold default",
                    min_value=0.30,
                    max_value=0.99,
                    value=float(base_mode.match_threshold),
                    step=0.01,
                )
                custom_detection_conf = st.slider(
                    "Detection confidence default",
                    min_value=0.10,
                    max_value=0.90,
                    value=float(base_mode.detection_confidence),
                    step=0.05,
                )
                custom_max_frames = st.number_input(
                    "Max frames default",
                    min_value=1,
                    max_value=100000,
                    value=int(base_mode.max_frames),
                    step=50,
                )

            st.write("Football feature flags")
            flag_col_1, flag_col_2 = st.columns(2)
            with flag_col_1:
                enable_ball_tracking = st.checkbox("Prepare ball tracking", value=base_mode.enable_ball_tracking)
                enable_pitch_mapping = st.checkbox("Prepare pitch mapping", value=base_mode.enable_pitch_mapping)
            with flag_col_2:
                enable_team_classification = st.checkbox(
                    "Prepare team classification",
                    value=base_mode.enable_team_classification,
                )
                enable_stats_aggregation = st.checkbox(
                    "Prepare stats aggregation",
                    value=base_mode.enable_stats_aggregation,
                )

            submitted = st.form_submit_button("Save custom mode")

        if submitted:
            try:
                custom_mode_id = normalize_mode_id(custom_mode_id_raw)
                custom_mode = ModeConfig(
                    mode_id=custom_mode_id,
                    name=custom_name.strip() or custom_mode_id,
                    description=custom_description.strip(),
                    pipeline_type=custom_pipeline_type,
                    yolo_model=custom_yolo_model,
                    tracker=custom_tracker,
                    encoder_backend=custom_encoder,
                    match_threshold=float(custom_threshold),
                    detection_confidence=float(custom_detection_conf),
                    image_size=base_mode.image_size,
                    reid_every_n_frames=base_mode.reid_every_n_frames,
                    min_good_frames_before_reid=base_mode.min_good_frames_before_reid,
                    min_embedding_quality=base_mode.min_embedding_quality,
                    min_update_quality=base_mode.min_update_quality,
                    max_frames=int(custom_max_frames),
                    min_crop_height=base_mode.min_crop_height,
                    min_crop_width=base_mode.min_crop_width,
                    crop_padding=base_mode.crop_padding,
                    device=base_mode.device,
                    draw_debug=base_mode.draw_debug,
                    live_preview_every_n_frames=base_mode.live_preview_every_n_frames,
                    enable_motion_analysis=base_mode.enable_motion_analysis,
                    draw_motion_vectors=base_mode.draw_motion_vectors,
                    motion_max_jump_fraction=base_mode.motion_max_jump_fraction,
                    motion_smoothing_alpha=base_mode.motion_smoothing_alpha,
                    motion_min_displacement_px=base_mode.motion_min_displacement_px,
                    enable_ball_tracking=bool(enable_ball_tracking),
                    enable_pitch_mapping=bool(enable_pitch_mapping),
                    enable_team_classification=bool(enable_team_classification),
                    enable_stats_aggregation=bool(enable_stats_aggregation),
                    is_custom=True,
                )
                saved = save_custom_mode(custom_mode, paths=paths, overwrite=False)
                st.success(f"Custom mode saved: {saved.name} ({saved.mode_id})")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


st.set_page_config(page_title="Local Person ReID MVP", layout="wide")

paths = AppPaths()
paths.ensure()

st.title("Local Person Re-Identification MVP")
st.caption("Lokales Demo-Setup mit auswählbaren Modi, YOLO Tracking, ReID Embeddings und SQLite Vector Store")

modes = list_modes(paths)

with st.sidebar:
    st.header("Mode")
    selected_mode_id = st.selectbox(
        "Analysis mode",
        options=list(modes.keys()),
        format_func=lambda mode_id: f"{modes[mode_id].name} ({mode_id})",
    )
    selected_mode = modes[selected_mode_id]
    st.caption(selected_mode.description)

    if selected_mode.pipeline_type == "football_analysis":
        st.info(
            "Football Mode v1 nutzt aktuell noch die stabile Default-ReID-Pipeline. "
            "Balltracking, Teamklassifikation, Pitch Mapping und Statistiken sind vorbereitet, aber noch nicht vollständig verdrahtet."
        )

    render_create_mode_form(paths, modes)

    st.divider()
    st.header("Pipeline Settings")

    input_type = st.radio("Input type", ["Video upload", "Local webcam"], index=0)
    yolo_model = st.text_input("YOLO model", value=selected_mode.yolo_model)
    tracker = st.selectbox(
        "Tracker",
        ["bytetrack.yaml", "botsort.yaml"],
        index=option_index(["bytetrack.yaml", "botsort.yaml"], selected_mode.tracker),
    )
    encoder_backend = st.selectbox(
        "Encoder backend",
        ["colorhist", "torchreid"],
        index=option_index(["colorhist", "torchreid"], selected_mode.encoder_backend),
    )

    match_threshold = st.slider(
        "Match threshold",
        min_value=0.30,
        max_value=0.99,
        value=float(selected_mode.match_threshold),
        step=0.01,
    )
    detection_confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=float(selected_mode.detection_confidence),
        step=0.05,
    )
    reid_every_n_frames = st.number_input(
        "ReID every N frames",
        min_value=1,
        max_value=100,
        value=int(selected_mode.reid_every_n_frames),
        step=1,
    )
    min_good_frames_before_reid = st.number_input(
        "Min good frames before first ReID match",
        min_value=1,
        max_value=20,
        value=int(selected_mode.min_good_frames_before_reid),
        step=1,
        help="Neue Tracks werden erst gespeichert oder gematcht, wenn genug hochwertige Crops gesammelt wurden.",
    )
    min_embedding_quality = st.slider(
        "Min crop quality for ReID candidates",
        min_value=0.00,
        max_value=1.00,
        value=float(selected_mode.min_embedding_quality),
        step=0.05,
        help="Crops unter diesem Qualitätswert werden nicht encodiert und nicht als neue ReID-Kandidaten genutzt.",
    )
    min_update_quality = st.slider(
        "Min crop quality for person embedding updates",
        min_value=0.00,
        max_value=1.00,
        value=float(selected_mode.min_update_quality),
        step=0.05,
        help="Bestehende Personen-Embeddings werden nur mit Crops ab diesem Qualitätswert aktualisiert.",
    )
    max_frames = st.number_input(
        "Max frames",
        min_value=1,
        max_value=100000,
        value=int(selected_mode.max_frames),
        step=50,
    )
    image_size = st.selectbox(
        "Image size",
        [320, 480, 640, 960, 1280],
        index=option_index([320, 480, 640, 960, 1280], selected_mode.image_size, fallback=2),
    )
    device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=option_index(["auto", "cpu", "cuda"], selected_mode.device))

    st.divider()
    st.header("Motion Analysis")
    enable_motion_analysis = st.checkbox(
        "Enable movement direction analysis",
        value=bool(selected_mode.enable_motion_analysis),
        help="Berechnet pro Track Bewegungsrichtung, Geschwindigkeit und Sprung-Plausibilität im Bildraum.",
    )
    draw_motion_vectors = st.checkbox(
        "Draw movement arrows",
        value=bool(selected_mode.draw_motion_vectors),
        disabled=not enable_motion_analysis,
        help="Zeichnet Richtungspfeile im Live-/Output-Bild. Rote Boxen markieren große Sprünge.",
    )
    motion_max_jump_fraction = st.slider(
        "Max jump fraction",
        min_value=0.05,
        max_value=0.80,
        value=float(selected_mode.motion_max_jump_fraction),
        step=0.05,
        disabled=not enable_motion_analysis,
        help="Maximal tolerierte Track-Verschiebung relativ zur Bilddiagonale. Größere Sprünge werden als auffällig markiert.",
    )
    motion_smoothing_alpha = st.slider(
        "Motion smoothing",
        min_value=0.00,
        max_value=1.00,
        value=float(selected_mode.motion_smoothing_alpha),
        step=0.05,
        disabled=not enable_motion_analysis,
        help="Glättung der Richtungspfeile. Höher = reagiert schneller, niedriger = ruhiger.",
    )

    st.divider()
    st.header("Live Display")
    show_live_preview = st.checkbox("Show live annotated preview", value=True)
    preview_every_n_frames = st.number_input(
        "Preview every N frames",
        min_value=1,
        max_value=30,
        value=int(selected_mode.live_preview_every_n_frames),
        step=1,
    )
    preview_width = st.slider("Preview width", min_value=480, max_value=1400, value=960, step=40)

    st.divider()
    st.caption("Datenschutz-Hinweis: Das MVP nutzt synthetische IDs und sollte nur mit berechtigtem/consented Material getestet werden.")

source: str | int | CameraSource | None = None
uploaded_file = None
selected_camera_source: CameraSource | None = None

if input_type == "Video upload":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_file is not None:
        input_path = ensure_unique_path(paths.input_dir / uploaded_file.name)
        input_path.write_bytes(uploaded_file.getbuffer())
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
        f"Run {selected_mode.name}",
        type="primary",
        width="stretch",
        disabled=source is None,
    )

with col_db:
    store = SQLiteVectorStore(paths.db_path)
    st.metric("Known synthetic persons", store.count_persons())

live_preview_placeholder = st.empty()
status_placeholder = st.empty()

if run_clicked and source is not None:
    config = selected_mode.to_pipeline_config(
        yolo_model=yolo_model,
        tracker=tracker,
        encoder_backend=encoder_backend,
        match_threshold=float(match_threshold),
        detection_confidence=float(detection_confidence),
        image_size=int(image_size),
        reid_every_n_frames=int(reid_every_n_frames),
        min_good_frames_before_reid=int(min_good_frames_before_reid),
        min_embedding_quality=float(min_embedding_quality),
        min_update_quality=float(min_update_quality),
        max_frames=int(max_frames),
        device=device,
        live_preview_every_n_frames=int(preview_every_n_frames),
        enable_motion_analysis=bool(enable_motion_analysis),
        draw_motion_vectors=bool(draw_motion_vectors),
        motion_max_jump_fraction=float(motion_max_jump_fraction),
        motion_smoothing_alpha=float(motion_smoothing_alpha),
    )

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

    try:
        pipeline = PersonReIdPipeline(config=config, paths=paths)
        result = pipeline.process(source, progress_callback=update_progress, frame_callback=update_live_preview)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.success(f"Finished mode {result.mode_name}. Processed {result.processed_frames} frames.")

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

store = SQLiteVectorStore(paths.db_path)
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
