from __future__ import annotations

import sys
import subprocess
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
from app.storage.store_factory import build_vector_store
from app.utils.camera_utils import (
    CameraSource,
    camera_backends,
    read_single_preview_frame,
    scan_local_cameras,
)
from app.utils.detail_utils import registry_groups_payload
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


def apply_polished_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --accent: #5b8cff;
            --accent-soft: rgba(91, 140, 255, 0.16);
            --panel: #171b24;
            --panel-2: #202532;
            --text-muted: #aab2c0;
        }
        .stApp { background: linear-gradient(180deg, #0f131b 0%, #111620 100%); }
        section[data-testid="stSidebar"] { background: #171b24; border-right: 1px solid #2a3140; }
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            letter-spacing: .01em;
        }
        div[data-testid="stMetric"] {
            background: #171b24;
            border: 1px solid #2d3545;
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }
        div[data-testid="stExpander"] {
            background: rgba(32,37,50,0.72);
            border: 1px solid #2d3545;
            border-radius: 12px;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(90deg, #5b8cff, #6f9cff);
            border: 0;
            color: #ffffff;
        }
        .stButton > button { border-radius: 10px; }
        .stSlider [data-baseweb="slider"] > div { color: #5b8cff; }
        .small-note { color: var(--text-muted); font-size: 0.9rem; }
        .mode-card {
            border: 1px solid #2d3545;
            background: linear-gradient(180deg, rgba(32,37,50,.85), rgba(23,27,36,.85));
            border-radius: 14px;
            padding: 1rem;
            margin-bottom: .8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def help_md(title: str, body: str) -> None:
    with st.expander(f"? {title}", expanded=False):
        st.caption(body)


def list_evaluation_runs(paths: AppPaths) -> list[Path]:
    root = paths.output_dir.parent / "evaluation_runs"
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)


def render_evaluation_results(paths: AppPaths) -> None:
    st.subheader("Evaluation results")
    st.caption("Liest all_videos_summary.csv aus data/evaluation_runs und zeigt die Testergebnisse filterbar im Default-Modus.")
    runs = list_evaluation_runs(paths)
    if not runs:
        st.info("Noch keine Evaluation-Runs gefunden. Erwarteter Ordner: data/evaluation_runs/<run_id>/all_videos_summary.csv")
        return
    run = st.selectbox("Evaluation run", runs, format_func=lambda p: p.name, help="Ordner mit all_videos_summary.csv auswählen.")
    summary_path = run / "all_videos_summary.csv"
    if not summary_path.exists():
        st.warning(f"Keine all_videos_summary.csv in {run}")
        return
    df = pd.read_csv(summary_path)
    if df.empty:
        st.info("Die Summary-Datei ist leer.")
        return

    with st.expander("Tabellenfilter und Spalten", expanded=True):
        cols = list(df.columns)
        selected_cols = st.multiselect("Angezeigte Spalten", cols, default=cols, help="Spalten ein-/ausblenden.")
        condition_col = next((c for c in cols if c.lower() in {"condition", "bedingung"}), None)
        if condition_col:
            conditions = sorted([str(x) for x in df[condition_col].dropna().unique()])
            selected_conditions = st.multiselect("Bedingungen", conditions, default=conditions)
            df = df[df[condition_col].astype(str).isin(selected_conditions)]
        search = st.text_input("Suche", value="", help="Filtert alle Zeilen über eine Volltextsuche in der Tabelle.")
        if search:
            mask = df.astype(str).apply(lambda row: row.str.contains(search, case=False, na=False).any(), axis=1)
            df = df[mask]
    if selected_cols:
        df = df[selected_cols]
    st.dataframe(df, width="stretch", hide_index=True)
    with st.expander("Wie sind diese Werte zu lesen?", expanded=False):
        st.markdown(
            """
            - **Track IDs**: temporäre IDs innerhalb eines Videos. Viele Track IDs deuten auf Tracking-Fragmentierung hin.
            - **Person IDs**: globale ReID-Profile. Mehrere IDs bei einer echten Person deuten auf ReID-Fragmentierung hin.
            - **Dominant Ratio / Expected Ratio**: Anteil der dominanten bzw. erwarteten Person-ID an allen ReID-Events.
            - **Neue Personen**: zeigt, wie oft das System trotz vorhandener Profile neue IDs erzeugt hat.
            - **Bewertung**: zusammenfassende Einstufung aus den berechneten Metriken.
            """
        )


def render_evaluation_runner(paths: AppPaths) -> None:
    with st.expander("Evaluation/Testlauf starten", expanded=False):
        st.caption("Startet scripts/evaluate_single_person_videos.py direkt aus der Oberfläche. Für lange Testläufe kann PowerShell weiterhin robuster sein.")
        manifest = st.text_input("Manifest", value=str(paths.input_dir.parent / "evaluation_manifest.csv"), help="CSV mit phase, video_path, condition, expected_person_id, notes.")
        output_dir = st.text_input("Output directory", value=str(paths.input_dir.parent / "evaluation_runs"), help="Hier entstehen all_videos_summary.csv und die Video-Unterordner.")
        test_mode = st.selectbox(
            "Test mode",
            ["adaptive_calibration", "fixed_db", "growing_db"],
            index=0,
            help="adaptive_calibration nutzt ein Kalibrierungsvideo als Start und erlaubt kontrolliertes Wachstum.",
        )
        max_frames_eval = st.number_input("Max frames für Test", min_value=0, max_value=100000, value=0, step=100, help="0 bedeutet komplettes Video.")
        cmd = [
            sys.executable,
            "scripts/evaluate_single_person_videos.py",
            "run",
            "--manifest", manifest,
            "--output-dir", output_dir,
            "--test-mode", test_mode,
            "--max-frames", str(int(max_frames_eval)),
        ]
        st.code(" ".join(cmd), language="powershell")
        if st.button("Testlauf ausführen", width="stretch"):
            try:
                proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=None)
                if proc.stdout:
                    st.text_area("stdout", proc.stdout[-12000:], height=260)
                if proc.stderr:
                    st.text_area("stderr", proc.stderr[-6000:], height=180)
                if proc.returncode == 0:
                    st.success("Testlauf abgeschlossen.")
                else:
                    st.error(f"Testlauf beendet mit Code {proc.returncode}.")
            except Exception as exc:
                st.error(f"Testlauf konnte nicht gestartet werden: {exc}")


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
                custom_vector_store_backend = st.selectbox(
                    "Vector store default",
                    options=["sqlite", "qdrant"],
                    index=option_index(["sqlite", "qdrant"], base_mode.vector_store_backend),
                )
                custom_qdrant_collection = st.text_input(
                    "Qdrant collection default",
                    value=base_mode.qdrant_collection,
                    disabled=custom_vector_store_backend != "qdrant",
                )
                custom_qdrant_mode = st.selectbox(
                    "Qdrant mode default",
                    options=["local", "server", "memory"],
                    index=option_index(["local", "server", "memory"], base_mode.qdrant_mode),
                    disabled=custom_vector_store_backend != "qdrant",
                    help="local läuft ohne Docker/Server und speichert auf Platte. server erwartet einen laufenden Qdrant-Dienst.",
                )
                custom_qdrant_local_path = st.text_input(
                    "Qdrant local path default",
                    value=base_mode.qdrant_local_path,
                    disabled=custom_vector_store_backend != "qdrant" or custom_qdrant_mode != "local",
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
                    vector_store_backend=custom_vector_store_backend,
                    qdrant_url=base_mode.qdrant_url,
                    qdrant_api_key=base_mode.qdrant_api_key,
                    qdrant_collection=custom_qdrant_collection,
                    qdrant_mode=custom_qdrant_mode,
                    qdrant_local_path=custom_qdrant_local_path,
                    qdrant_prefer_grpc=base_mode.qdrant_prefer_grpc,
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
                    disable_internal_motion_when_botsort=base_mode.disable_internal_motion_when_botsort,
                    motion_max_jump_fraction=base_mode.motion_max_jump_fraction,
                    motion_smoothing_alpha=base_mode.motion_smoothing_alpha,
                    motion_min_displacement_px=base_mode.motion_min_displacement_px,
                    enable_detail_analysis=base_mode.enable_detail_analysis,
                    detail_weight=base_mode.detail_weight,
                    detail_min_confidence=base_mode.detail_min_confidence,
                    draw_detail_labels=base_mode.draw_detail_labels,
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
apply_polished_theme()

paths = AppPaths()
paths.ensure()

st.title("Local Person Re-Identification MVP")
st.caption("Lokales Demo-Setup mit auswählbaren Modi, YOLO Tracking, ReID Embeddings und SQLite/Qdrant Vector Store")

modes = list_modes(paths)

with st.sidebar:
    st.header("Mode")
    selected_mode_id = st.selectbox(
        "Analysis mode",
        options=list(modes.keys()),
        format_func=lambda mode_id: f"{modes[mode_id].name} ({mode_id})",
    )
    selected_mode = modes[selected_mode_id]
    st.markdown(
        f"""<div class='mode-card'><b>{selected_mode.name}</b><br>
        <span class='small-note'>{selected_mode.description}</span><br>
        <span class='small-note'>Pipeline: {selected_mode.pipeline_type} · Tracker: {selected_mode.tracker} · Encoder: {selected_mode.encoder_backend}</span></div>""",
        unsafe_allow_html=True,
    )

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
    vector_store_backend = st.selectbox(
        "Vector store",
        ["sqlite", "qdrant"],
        index=option_index(["sqlite", "qdrant"], selected_mode.vector_store_backend),
        help="SQLite ist die leichte MVP-Variante. Qdrant nutzt eine echte Vector Database für Embedding-Suche.",
    )
    qdrant_url = selected_mode.qdrant_url
    qdrant_collection = selected_mode.qdrant_collection
    qdrant_mode = selected_mode.qdrant_mode
    qdrant_local_path = selected_mode.qdrant_local_path
    qdrant_prefer_grpc = selected_mode.qdrant_prefer_grpc
    if vector_store_backend == "qdrant":
        qdrant_mode = st.selectbox(
            "Qdrant mode",
            ["local", "server", "memory"],
            index=option_index(["local", "server", "memory"], selected_mode.qdrant_mode),
            help="local läuft ohne Docker/Server und speichert in einem lokalen Ordner. server nutzt http://localhost:6333 oder eine Qdrant-Cloud/Server-URL.",
        )
        qdrant_collection = st.text_input("Qdrant collection", value=selected_mode.qdrant_collection)
        if qdrant_mode == "local":
            qdrant_local_path = st.text_input("Qdrant local path", value=selected_mode.qdrant_local_path)
            st.caption("Qdrant Local nutzt qdrant-client eingebettet im Python-Prozess. Es wird kein Docker und kein localhost:6333 benötigt.")
        elif qdrant_mode == "server":
            qdrant_url = st.text_input("Qdrant URL", value=selected_mode.qdrant_url)
            qdrant_prefer_grpc = st.checkbox("Prefer Qdrant gRPC", value=bool(selected_mode.qdrant_prefer_grpc))
        else:
            st.caption("Qdrant Memory speichert nur bis zum Neustart der App und ist nur für schnelle Tests sinnvoll.")

    parameter_profile = st.selectbox(
        "Parameter profile",
        ["Standard/Video", "Kurzvideo adaptiv", "Langes Video stabil", "Kalibrierung Qualität"],
        index=0,
        help="Setzt sinnvolle Startwerte. Einzelne Regler können danach weiter angepasst werden.",
    )

    with st.expander("Kalibrierung im normalen Projekt", expanded=False):
        st.caption(
            "Kalibrierung baut ein Personenprofil aus hochwertigen Crops auf. "
            "Sie ist nicht nur für Evaluation gedacht, sondern kann direkt in der normalen Pipeline genutzt werden."
        )
        calibration_ui_mode = st.radio(
            "Kalibrierungsmodus",
            ["Aus", "Neue Person kalibrieren", "Bestehende Person erweitern"],
            index=0,
            help="Neue Person: erstellt ein Startprofil. Bestehende Person: fügt hochwertige Embeddings zu einer vorhandenen person_id hinzu.",
        )
        calibration_label = st.text_input(
            "Kalibrierungsnotiz/Label",
            value="",
            help="Optionaler Hinweis wie 'weißes Shirt Allaround' oder 'Brille Profilupdate'.",
        )
        available_person_ids: list[str] = []
        try:
            tmp_config = selected_mode.to_pipeline_config(
                vector_store_backend=vector_store_backend,
                qdrant_url=qdrant_url,
                qdrant_api_key=selected_mode.qdrant_api_key,
                qdrant_collection=qdrant_collection,
                qdrant_mode=qdrant_mode,
                qdrant_local_path=qdrant_local_path,
                qdrant_prefer_grpc=bool(qdrant_prefer_grpc),
            )
            tmp_store = build_vector_store(tmp_config, paths)
            available_person_ids = [p.person_id for p in tmp_store.list_persons()]
            if hasattr(tmp_store, "close"):
                tmp_store.close()
        except Exception:
            available_person_ids = []
        calibration_target_person_id = ""
        if calibration_ui_mode == "Bestehende Person erweitern":
            if available_person_ids:
                calibration_target_person_id = st.selectbox(
                    "Zielperson",
                    available_person_ids,
                    help="Diese person_id wird mit hochwertigen Kalibrierungs-Crops erweitert.",
                )
            else:
                st.warning("Noch keine gespeicherte Person vorhanden. Erst eine neue Person kalibrieren oder ein normales Video laufen lassen.")
        st.info(
            "Kalibrierung nutzt strengere Qualitätswerte: höhere Auflösung, größere Mindest-Crops und weniger Detail-Einfluss. "
            "Das Profil wächst anschließend kontrolliert über die Three-Zone-Logik."
        )

    profile_values = {
        "Standard/Video": {
            "match_threshold": selected_mode.match_threshold,
            "strong": selected_mode.strong_match_threshold,
            "weak": selected_mode.weak_match_threshold,
            "detection": selected_mode.detection_confidence,
            "image_size": selected_mode.image_size,
            "reid_every": selected_mode.reid_every_n_frames,
            "min_good": selected_mode.min_good_frames_before_reid,
            "min_embed": selected_mode.min_embedding_quality,
            "min_update": selected_mode.min_update_quality,
            "crop_h": selected_mode.min_crop_height,
            "crop_w": selected_mode.min_crop_width,
            "padding": selected_mode.crop_padding,
            "detail_weight": selected_mode.detail_weight,
            "detail_conf": selected_mode.detail_min_confidence,
            "detail_enabled": selected_mode.enable_detail_analysis,
        },
        "Kurzvideo adaptiv": {
            "match_threshold": 0.72, "strong": 0.80, "weak": 0.66, "detection": 0.35,
            "image_size": 960, "reid_every": 3, "min_good": 3, "min_embed": 0.58, "min_update": 0.75,
            "crop_h": 110, "crop_w": 40, "padding": 0.08, "detail_weight": 0.03, "detail_conf": 0.70, "detail_enabled": True,
        },
        "Langes Video stabil": {
            "match_threshold": 0.76, "strong": 0.82, "weak": 0.68, "detection": 0.40,
            "image_size": 960, "reid_every": 5, "min_good": 4, "min_embed": 0.60, "min_update": 0.78,
            "crop_h": 120, "crop_w": 45, "padding": 0.08, "detail_weight": 0.05, "detail_conf": 0.70, "detail_enabled": True,
        },
        "Kalibrierung Qualität": {
            "match_threshold": 0.78, "strong": 0.84, "weak": 0.70, "detection": selected_mode.calibration_detection_confidence,
            "image_size": selected_mode.calibration_image_size, "reid_every": selected_mode.calibration_reid_every_n_frames,
            "min_good": selected_mode.calibration_min_good_frames_before_reid, "min_embed": selected_mode.calibration_min_embedding_quality,
            "min_update": selected_mode.calibration_min_update_quality, "crop_h": selected_mode.calibration_min_crop_height,
            "crop_w": selected_mode.calibration_min_crop_width, "padding": selected_mode.calibration_crop_padding,
            "detail_weight": 0.0, "detail_conf": 0.75, "detail_enabled": selected_mode.calibration_enable_detail_analysis,
        },
    }[parameter_profile]
    if calibration_ui_mode != "Aus":
        profile_values = {
            "match_threshold": 0.78,
            "strong": 0.84,
            "weak": 0.70,
            "detection": selected_mode.calibration_detection_confidence,
            "image_size": selected_mode.calibration_image_size,
            "reid_every": selected_mode.calibration_reid_every_n_frames,
            "min_good": selected_mode.calibration_min_good_frames_before_reid,
            "min_embed": selected_mode.calibration_min_embedding_quality,
            "min_update": selected_mode.calibration_min_update_quality,
            "crop_h": selected_mode.calibration_min_crop_height,
            "crop_w": selected_mode.calibration_min_crop_width,
            "padding": selected_mode.calibration_crop_padding,
            "detail_weight": 0.0,
            "detail_conf": 0.75,
            "detail_enabled": selected_mode.calibration_enable_detail_analysis,
        }

    match_threshold = st.slider(
        "Match threshold",
        min_value=0.30,
        max_value=0.99,
        value=float(profile_values["match_threshold"]),
        step=0.01,
    )
    with st.expander("Three-zone ReID decision", expanded=True):
        strong_match_threshold = st.slider(
            "Strong match threshold",
            min_value=0.50,
            max_value=0.99,
            value=float(profile_values["strong"]),
            step=0.01,
            help="Ab diesem Score wird eine bestehende Person sicher übernommen und darf wachsen.",
        )
        weak_match_threshold = st.slider(
            "Weak/pending match threshold",
            min_value=0.30,
            max_value=0.95,
            value=float(profile_values["weak"]),
            step=0.01,
            help="Ab diesem Score wird keine neue Person erzeugt, sondern ein unsicherer Kandidat gehalten.",
        )
        new_person_max_score = st.slider(
            "New person max score",
            min_value=0.20,
            max_value=0.90,
            value=float(selected_mode.new_person_max_score),
            step=0.01,
            help="Nur wenn Scores über mehrere Events höchstens in diesem Bereich liegen, darf eine neue Person entstehen.",
        )
        new_person_min_evidence_events = st.number_input(
            "New person min evidence events",
            min_value=1,
            max_value=30,
            value=int(selected_mode.new_person_min_evidence_events),
            step=1,
        )
        new_person_evidence_window_frames = st.number_input(
            "New person evidence window frames",
            min_value=5,
            max_value=300,
            value=int(selected_mode.new_person_evidence_window_frames),
            step=5,
        )
        new_person_low_match_ratio = st.slider(
            "New person low-match ratio",
            min_value=0.50,
            max_value=1.00,
            value=float(selected_mode.new_person_low_match_ratio),
            step=0.05,
        )

    detection_confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=float(profile_values["detection"]),
        step=0.05,
    )
    reid_every_n_frames = st.number_input(
        "ReID every N frames",
        min_value=1,
        max_value=100,
        value=int(profile_values["reid_every"]),
        step=1,
    )
    min_good_frames_before_reid = st.number_input(
        "Min good frames before first ReID match",
        min_value=1,
        max_value=20,
        value=int(profile_values["min_good"]),
        step=1,
        help="Neue Tracks werden erst gespeichert oder gematcht, wenn genug hochwertige Crops gesammelt wurden.",
    )
    min_embedding_quality = st.slider(
        "Min crop quality for ReID candidates",
        min_value=0.00,
        max_value=1.00,
        value=float(profile_values["min_embed"]),
        step=0.05,
        help="Crops unter diesem Qualitätswert werden nicht encodiert und nicht als neue ReID-Kandidaten genutzt.",
    )
    min_update_quality = st.slider(
        "Min crop quality for person embedding updates",
        min_value=0.00,
        max_value=1.00,
        value=float(profile_values["min_update"]),
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
        index=option_index([320, 480, 640, 960, 1280], int(profile_values["image_size"]), fallback=3),
    )
    with st.expander("Crop quality gate", expanded=False):
        min_crop_height = st.number_input(
            "Min crop height", min_value=40, max_value=600, value=int(profile_values["crop_h"]), step=10
        )
        min_crop_width = st.number_input(
            "Min crop width", min_value=20, max_value=300, value=int(profile_values["crop_w"]), step=5
        )
        crop_padding = st.slider(
            "Crop padding", min_value=0.00, max_value=0.25, value=float(profile_values["padding"]), step=0.01
        )

    device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=option_index(["auto", "cpu", "cuda"], selected_mode.device))

    st.divider()
    st.header("Motion Analysis")
    disable_internal_motion_when_botsort = st.checkbox(
        "Disable internal direction analysis when BoT-SORT is selected",
        value=bool(selected_mode.disable_internal_motion_when_botsort),
        help="Für die Max-Variante wird die eigene Richtungserkennung deaktiviert, sobald BoT-SORT genutzt wird.",
    )
    botsort_disables_internal_motion = disable_internal_motion_when_botsort and "botsort" in tracker.lower()
    if botsort_disables_internal_motion:
        st.caption("Eigene Direction-Erkennung ist in diesem Lauf deaktiviert, weil BoT-SORT aktiv ist.")

    enable_motion_analysis = st.checkbox(
        "Enable movement direction analysis",
        value=bool(selected_mode.enable_motion_analysis) and not botsort_disables_internal_motion,
        disabled=botsort_disables_internal_motion,
        help="Berechnet pro Track Bewegungsrichtung, Geschwindigkeit und Sprung-Plausibilität im Bildraum.",
    )
    draw_motion_vectors = st.checkbox(
        "Draw movement arrows",
        value=bool(selected_mode.draw_motion_vectors) and not botsort_disables_internal_motion,
        disabled=not enable_motion_analysis or botsort_disables_internal_motion,
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
    st.header("Detail Analysis")
    enable_detail_analysis = st.checkbox(
        "Enable detail recognition layer",
        value=bool(profile_values["detail_enabled"]),
        help="Extrahiert weiche Detail-Signale aus Person-Crops: Kopf, Oberkörper, Unterkörper, Kleidung, Farben, Accessoires und Muster. Details unterstützen OSNet nur als Re-Ranking-Signal.",
    )
    detail_weight = st.slider(
        "Detail matching weight",
        min_value=0.00,
        max_value=0.35,
        value=float(profile_values["detail_weight"]),
        step=0.01,
        disabled=not enable_detail_analysis,
        help="Kleine Werte sind sicherer. Details sollen OSNet unterstützen, aber nicht ersetzen.",
    )
    detail_min_confidence = st.slider(
        "Min detail confidence",
        min_value=0.30,
        max_value=0.90,
        value=float(profile_values["detail_conf"]),
        step=0.05,
        disabled=not enable_detail_analysis,
        help="Unterhalb dieses Werts wird ein Detail als unbekannt markiert.",
    )
    draw_detail_labels = st.checkbox(
        "Draw detail labels",
        value=bool(selected_mode.draw_detail_labels),
        disabled=not enable_detail_analysis,
        help="Zeigt erkannte Detailzustände direkt in der Bounding-Box-Beschriftung.",
    )
    with st.expander("Welche Details werden genutzt?"):
        registry_rows = []
        for group, specs in registry_groups_payload().items():
            for spec in specs:
                registry_rows.append(
                    {
                        "group": group,
                        "detail": spec["label"],
                        "technical_key": spec["name"],
                        "kind": spec["kind"],
                        "weight": spec["weight"],
                        "volatile": spec["volatile"],
                        "description": spec["description"],
                    }
                )
        st.dataframe(pd.DataFrame(registry_rows), width="stretch", hide_index=True)
        st.caption(
            "Volatile Details wie Kappe, Kapuze, Uhr oder Aufdruck werden absichtlich niedriger/weicher gewichtet, "
            "damit ein Wechsel dieser Details nicht automatisch die Person-ID zerstört."
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
    metric_config = selected_mode.to_pipeline_config(
        yolo_model=yolo_model,
        tracker=tracker,
        encoder_backend=encoder_backend,
        vector_store_backend=vector_store_backend,
        qdrant_url=qdrant_url,
        qdrant_api_key=selected_mode.qdrant_api_key,
        qdrant_collection=qdrant_collection,
        qdrant_mode=qdrant_mode,
        qdrant_local_path=qdrant_local_path,
        qdrant_prefer_grpc=bool(qdrant_prefer_grpc),
    )
    try:
        store = build_vector_store(metric_config, paths)
        st.metric("Known synthetic persons", store.count_persons())
        if hasattr(store, "close"):
            store.close()
    except Exception as exc:
        store = None
        st.warning(f"Vector store not reachable: {exc}")

live_preview_placeholder = st.empty()
status_placeholder = st.empty()

if run_clicked and source is not None:
    config = selected_mode.to_pipeline_config(
        yolo_model=yolo_model,
        tracker=tracker,
        encoder_backend=encoder_backend,
        vector_store_backend=vector_store_backend,
        qdrant_url=qdrant_url,
        qdrant_api_key=selected_mode.qdrant_api_key,
        qdrant_collection=qdrant_collection,
        qdrant_mode=qdrant_mode,
        qdrant_local_path=qdrant_local_path,
        qdrant_prefer_grpc=bool(qdrant_prefer_grpc),
        match_threshold=float(match_threshold),
        strong_match_threshold=float(strong_match_threshold),
        weak_match_threshold=float(weak_match_threshold),
        new_person_max_score=float(new_person_max_score),
        new_person_min_evidence_events=int(new_person_min_evidence_events),
        new_person_evidence_window_frames=int(new_person_evidence_window_frames),
        new_person_low_match_ratio=float(new_person_low_match_ratio),
        detection_confidence=float(detection_confidence),
        image_size=int(image_size),
        reid_every_n_frames=int(reid_every_n_frames),
        min_good_frames_before_reid=int(min_good_frames_before_reid),
        min_embedding_quality=float(min_embedding_quality),
        min_update_quality=float(min_update_quality),
        max_frames=int(max_frames),
        min_crop_height=int(min_crop_height),
        min_crop_width=int(min_crop_width),
        crop_padding=float(crop_padding),
        device=device,
        live_preview_every_n_frames=int(preview_every_n_frames),
        enable_motion_analysis=bool(enable_motion_analysis),
        draw_motion_vectors=bool(draw_motion_vectors),
        disable_internal_motion_when_botsort=bool(disable_internal_motion_when_botsort),
        motion_max_jump_fraction=float(motion_max_jump_fraction),
        motion_smoothing_alpha=float(motion_smoothing_alpha),
        enable_detail_analysis=bool(enable_detail_analysis),
        detail_weight=float(detail_weight),
        detail_min_confidence=float(detail_min_confidence),
        draw_detail_labels=bool(draw_detail_labels),
        calibration_mode=("new_person" if calibration_ui_mode == "Neue Person kalibrieren" else "extend_person" if calibration_ui_mode == "Bestehende Person erweitern" else "off"),
        calibration_target_person_id=str(calibration_target_person_id or ""),
        calibration_label=str(calibration_label or ""),
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
        try:
            result = pipeline.process(source, progress_callback=update_progress, frame_callback=update_live_preview)
        finally:
            if hasattr(pipeline, "close"):
                pipeline.close()
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

table_config = selected_mode.to_pipeline_config(
    yolo_model=yolo_model,
    tracker=tracker,
    encoder_backend=encoder_backend,
    vector_store_backend=vector_store_backend,
    qdrant_url=qdrant_url,
    qdrant_api_key=selected_mode.qdrant_api_key,
    qdrant_collection=qdrant_collection,
    qdrant_mode=qdrant_mode,
    qdrant_local_path=qdrant_local_path,
    qdrant_prefer_grpc=bool(qdrant_prefer_grpc),
)
try:
    store = build_vector_store(table_config, paths)
    persons_df = store.persons_dataframe()
    events_df = store.events_dataframe(limit=200)
    runs_df = store.analysis_runs_dataframe(limit=100)
    if hasattr(store, "close"):
        store.close()
except Exception as exc:
    st.warning(f"Could not load stored results: {exc}")
    persons_df = pd.DataFrame()
    events_df = pd.DataFrame()
    runs_df = pd.DataFrame()

with st.expander("Analysis runs", expanded=True):
    st.caption("Gespeicherte Verarbeitungsläufe. Quelle, FPS, Frame-Anzahl und Modus helfen, spätere Ergebnisse nachzuvollziehen.")
    help_md("Analysis runs", "Jede Zeile entspricht einem Video- oder Webcam-Lauf. run_id verbindet den Lauf mit Events, Personenupdates und Output-Videos.")
    if runs_df.empty:
        st.info("No analysis runs stored yet.")
    else:
        run_cols = st.multiselect("Spalten Analysis runs", list(runs_df.columns), default=list(runs_df.columns), key="runs_cols")
        st.dataframe(runs_df[run_cols] if run_cols else runs_df, width="stretch", hide_index=True)

with st.expander("Stored synthetic persons", expanded=False):
    st.caption("Langfristige globale Personenprofile. Beobachtungen zeigen, wie stark ein Profil bereits gewachsen ist.")
    help_md("Stored synthetic persons", "person_id ist eine synthetische ID. observations zählt gespeicherte hochwertige Embedding-Updates. Viele Beobachtungen können die ReID stabilisieren, wenn die Updates korrekt zugeordnet wurden.")
    if persons_df.empty:
        st.info("No persons stored yet.")
    else:
        person_cols = st.multiselect("Spalten Personen", list(persons_df.columns), default=list(persons_df.columns), key="person_cols")
        st.dataframe(persons_df[person_cols] if person_cols else persons_df, width="stretch", hide_index=True)

with st.expander("Recent events", expanded=False):
    st.caption("Frame-/Track-/ReID-Ereignisse. Diese Tabelle erklärt, warum Personen erstellt, gematcht oder nur als Pending behandelt wurden.")
    help_md("Recent events", "Wichtige Felder: event_type beschreibt die Entscheidung; match_score ist der finale ReID-Score; decision_zone zeigt strong/weak/low/calibration; quality_score bewertet den Crop; top_matches enthält die besten Kandidaten.")
    if events_df.empty:
        st.info("No events stored yet.")
    else:
        event_cols_default = [c for c in ["event_id", "event_type", "person_id", "frame_index", "track_id", "score", "quality_score", "match_visual_score", "match_detail_score", "match_reason", "created_at"] if c in events_df.columns]
        event_cols = st.multiselect("Spalten Events", list(events_df.columns), default=event_cols_default or list(events_df.columns), key="event_cols")
        event_search = st.text_input("Event-Suche", value="", key="event_search")
        event_view = events_df.copy()
        if event_search:
            mask = event_view.astype(str).apply(lambda row: row.str.contains(event_search, case=False, na=False).any(), axis=1)
            event_view = event_view[mask]
        st.dataframe(event_view[event_cols] if event_cols else event_view, width="stretch", hide_index=True)

if selected_mode_id == "default":
    st.divider()
    render_evaluation_results(paths)
    render_evaluation_runner(paths)
