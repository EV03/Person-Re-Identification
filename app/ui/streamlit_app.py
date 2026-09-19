"""Streamlit entry point for configuring, running and inspecting analyses.

Streamlit executes this module from top to bottom on every interaction.  Values
that must survive a rerun therefore belong in ``st.session_state`` or a cache;
pipeline work should only start inside an explicit user-action branch.
"""

from __future__ import annotations

import json
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
from app.storage.encoder_paths import paths_for_encoder
from app.evaluation.runner import create_unit_paths
from app.evaluation.single_person import create_single_person_report, load_main_run_metrics
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
from app.utils.model_discovery import (
    OSNET_MODEL_NAMES,
    discover_reid_models,
    discover_yolo_models,
    display_labels,
    reid_architectures,
    selectable_paths,
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
    yolo_models = discover_yolo_models()
    yolo_labels = display_labels(yolo_models)
    st.selectbox(
        "YOLO model",
        options=selectable_paths(st.session_state["pipeline_yolo_model"], yolo_models),
        format_func=lambda value: yolo_labels.get(value, value),
        key="pipeline_yolo_model",
        accept_new_options=True,
        help="Lokal gefundene Gewichte werden angeboten; ein eigener Pfad kann weiterhin eingegeben werden.",
    )
    st.text_input(
        "Tracker configuration", key="pipeline_tracker",
        help="bytetrack.yaml, botsort.yaml oder eine eigene YAML-Datei. Interne Tracker-Schwellen werden in dieser Datei eingestellt.",
    )
    st.selectbox("Encoder backend", ["colorhist", "torchreid"], key="pipeline_encoder_backend")
    reid_models = discover_reid_models()
    architecture_options = list(dict.fromkeys([
        st.session_state["pipeline_reid_model_name"],
        *reid_architectures(reid_models),
        *OSNET_MODEL_NAMES,
    ]))
    st.selectbox(
        "ReID model",
        options=architecture_options,
        key="pipeline_reid_model_name",
        accept_new_options=True,
        help="Architektur des ReID-Encoders. Erkannte OSNet-Architekturen lokaler Checkpoints stehen zuerst.",
    )
    reid_labels = display_labels(reid_models)
    st.selectbox(
        "ReID checkpoint",
        options=selectable_paths(st.session_state["pipeline_reid_checkpoint"], reid_models),
        format_func=lambda value: reid_labels.get(value, value),
        key="pipeline_reid_checkpoint",
        accept_new_options=True,
        help="Explizite ReID-Gewichte für OSNet. Relative Pfade beziehen sich auf das Projekt; Farbhistogramme ignorieren dieses Feld.",
    )
    st.text_input("Device", key="pipeline_device", help="auto, cpu, cuda oder cuda:0")
    st.selectbox(
        "Identity decision policy",
        options=["main_single_threshold", "details_tracking_v2"],
        format_func=lambda value: {
            "main_single_threshold": "Main: einzelne Match-Schwelle",
            "details_tracking_v2": "Details Tracking: Detail-/Evidenz-Pipeline",
        }[value],
        key="pipeline_decision_policy",
        help="Dieser Algorithmus gehört zum Preset und ist unabhängig vom unten gewählten Testmodul.",
    )

    st.subheader("Schwellenwerte")
    for field, label, lower, upper, help_text in (
        ("match_threshold", "Match threshold", -1.0, 1.0, "Cosine Similarity: Ein Profil wird ab diesem Wert akzeptiert."),
        # Ultralytics' track() replaces an exact zero with its 0.1 default.
        ("detection_confidence", "Detection confidence", 0.0001, 1.0, "Konfidenzgrenze für YOLO. Muss positiv sein: Ultralytics würde exakt 0 intern durch 0,1 ersetzen."),
        ("min_embedding_quality", "Min crop quality for ReID candidates", 0.0, 1.0, "0 deaktiviert diese Qualitätsschwelle; Mindestgrößen und Qualitätsgewichtung bleiben erhalten."),
        ("min_initial_blur_score", "Min. Schärfe für Initialkandidaten", 0.0, 1.0, "Unbekannte Tracks sammeln nur Crops ab diesem Schärfewert. 0 deaktiviert die separate Schärfegrenze."),
        ("min_border_blur_score", "Min. Schärfe bei Randkontakt", 0.0, 1.0, "Strengere Schärfegrenze, wenn die Personenbox einen Bildrand berührt. Eine scharfe Seitenansicht bleibt erlaubt."),
        ("min_update_quality", "Min crop quality for person embedding updates", 0.0, 1.0, "Updates müssen zusätzlich die Kandidatenschwelle erfüllen."),
        ("min_update_similarity", "Min similarity for person embedding updates", -1.0, 1.0, "Zusätzlicher Profilschutz: Das neue Embedding muss zum bestehenden Personenprofil passen. -1 lässt alle gültigen Ähnlichkeiten zu. Auf Pilotclips abstimmen."),
        ("max_person_overlap_ratio", "Max person overlap for ReID", 0.0, 1.0, "Maximal erlaubter Schnittflächenanteil relativ zur kleineren Personenbox. 1 deaktiviert die Überlappungssperre."),
    ):
        st.number_input(label, min_value=lower, max_value=upper, step=0.01,
                        format="%.4f", key=f"pipeline_{field}", help=help_text)

    st.subheader("Ausschnitte und zeitliche Parameter")
    for field, label, minimum, help_text in (
        ("min_crop_width", "Min crop width (px)", 0, "0 deaktiviert die Mindestbreite; leere Crops bleiben ungültig."),
        ("min_crop_height", "Min crop height (px)", 0, "0 deaktiviert die Mindesthöhe; leere Crops bleiben ungültig."),
        ("min_good_frames_before_reid", "Min good frames before first ReID match", 1, "Anzahl akzeptierter Beobachtungen vor der ersten Identitätsentscheidung."),
        ("initial_candidate_every_n_frames", "Initial candidate every N frames", 1, "Mindestabstand zwischen akzeptierten Initialbeobachtungen eines unbekannten Tracks."),
        ("reid_every_n_frames", "ReID update every N frames", 1, "Update-Intervall bereits zugeordneter Tracks."),
        ("image_size", "Image size", 32, "YOLO-Eingangsgröße; Vielfache von 32 verwenden."),
        ("max_frames", "Max frames (0 = vollständiges Video)", 0, "Begrenzt hochgeladene Videos. Für Webcam-Läufe gilt die separat eingestellte Aufnahmedauer."),
        ("overlap_cooldown_frames", "Overlap cooldown (frames)", 0, "Nach einer starken Personenüberlappung werden so viele Frames lang keine ReID-Profile angelegt oder aktualisiert."),
        ("track_state_ttl_frames", "Track state TTL (frames)", 1, "Nach so vielen fehlenden Frames wird die laufinterne Zuordnung eines verschwundenen Tracks verworfen. Bei einer Rückkehr ist dadurch eine neue ReID-Entscheidung erforderlich."),
    ):
        st.number_input(label, min_value=minimum, step=1, key=f"pipeline_{field}", help=help_text)
    st.number_input("Crop padding", min_value=0.0, step=0.01, format="%.4f",
                    key="pipeline_crop_padding", help="Zusätzlicher Rand relativ zur Boxgröße, z. B. 0,05 = 5 % pro Seite.")

    with st.expander("Details-Tracking-Methodik"):
        if st.session_state["pipeline_decision_policy"] != "details_tracking_v2":
            st.caption("Diese Werte werden gespeichert, aber erst mit der Details-Tracking-Policy aktiv.")
        for field, label, help_text in (
            ("detail_reranking_enabled", "Detail-Re-Ranking aktiv", "Extrahiert die Detail-Registry und kombiniert sie mit dem visuellen ReID-Score."),
            ("weak_match_zone_enabled", "Weak-Zone aktiv", "Erlaubt vorläufige Zuordnungen ohne Profilupdate zwischen Weak- und Strong-Schwelle."),
            ("delayed_new_person_enabled", "Neue IDs verzögern", "Sammelt wiederholte Low-Evidenz, bevor ein neues Personenprofil angelegt wird."),
            ("overlap_protection_enabled", "Überlappungsschutz aktiv", "Unterdrückt neue IDs für stark überlappende Low-Kandidaten."),
            ("motion_continuity_enabled", "Motion-Kontinuitätsbonus aktiv", "Erlaubt einen begrenzten Bonus für räumlich plausible Fortsetzungen."),
        ):
            st.checkbox(label, key=f"pipeline_{field}", help=help_text)
        for field, label, help_text in (
            ("detail_weight", "Detail reranking weight", "Gewicht visueller Details zusätzlich zur ReID-Cosine-Similarity."),
            ("detail_min_confidence", "Minimum detail confidence", "Mindestvertrauen, ab dem extrahierte Details das Ranking beeinflussen."),
            ("new_person_low_match_ratio", "Required low-match ratio", "Anteil niedriger Scores im Evidenzfenster für die verzögerte Neuanlage."),
            ("new_person_overlap_threshold", "Overlap protection threshold", "Verhindert neue IDs bei zu starker zeitlicher Überlappung mit bekannten Identitäten."),
            ("motion_identity_bonus", "Motion continuity bonus", "Kleiner Bonus für räumlich plausible Fortsetzungen derselben Identität."),
            ("motion_identity_max_distance_fraction", "Motion maximum distance fraction", "Maximaler Mittelpunktabstand relativ zur Bilddiagonale für den Bonus."),
        ):
            st.number_input(label, min_value=0.0, max_value=1.0, step=0.01, format="%.4f",
                            key=f"pipeline_{field}", help=help_text)
        for field, label, help_text in (
            ("strong_match_threshold", "Strong match threshold", "Ab diesem kombinierten Score wird die Identität sicher übernommen und aktualisiert."),
            ("weak_match_threshold", "Weak match threshold", "Ab diesem Score bleibt eine Zuordnung vorläufig; das Personenprofil wird nicht aktualisiert."),
            ("new_person_max_score", "New-person maximum score", "Nur Kandidaten unterhalb dieses Scores liefern Evidenz für eine neue Person."),
        ):
            st.number_input(label, min_value=-1.0, max_value=1.0, step=0.01, format="%.4f",
                            key=f"pipeline_{field}", help=help_text)
        for field, label, help_text in (
            ("new_person_min_evidence_events", "Minimum new-person evidence events", "Benötigte niedrige Match-Beobachtungen vor einer neuen synthetischen ID."),
            ("new_person_min_evidence_span_frames", "Minimum evidence span (frames)", "Mindestdauer der Evidenz, damit kurzzeitige Fehler keine neue ID erzeugen."),
            ("new_person_evidence_window_frames", "Evidence window (frames)", "Zeitfenster, in dem die Evidenz gesammelt wird."),
            ("motion_identity_max_frame_gap", "Motion maximum frame gap", "Maximaler Abstand zur letzten Beobachtung für den Kontinuitätsbonus."),
        ):
            st.number_input(label, min_value=1, step=1, key=f"pipeline_{field}", help=help_text)

    st.subheader("Ausgabeparameter")
    st.checkbox("Annotate output video", key="pipeline_draw_debug")
    st.number_input("Preview every N frames", min_value=1, step=1,
                    key="pipeline_live_preview_every_n_frames")
    return {field: st.session_state[f"pipeline_{field}"] for field in RUNTIME_PARAMETER_FIELDS}


def queue_saved_preset(saved: ModeConfig, message: str) -> None:
    """Select a saved preset on the next rerun and show one success notice."""
    st.session_state["pending_preset_id"] = saved.mode_id
    st.session_state["preset_saved_notice"] = message
    st.rerun()


def render_preset_forms(paths: AppPaths, config: PipelineConfig, selected_mode: ModeConfig) -> None:
    with st.expander("Aktuelle Einstellungen als neue Versuchskonfiguration speichern"):
        st.caption("Speichert exakt alle oben eingestellten Pipeline-Parameter unter einer neuen ID. B0/A1/A2/A3/D1-D6 und vorhandene Presets werden nicht überschrieben.")
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
                queue_saved_preset(saved, f"Gespeichert und geladen: {saved.name} ({saved.mode_id})")

    if selected_mode.is_custom:
        with st.expander("Ausgewählte Versuchskonfiguration aktualisieren"):
            st.caption(
                f"Überschreibt '{selected_mode.mode_id}' mit allen aktuell eingestellten Parametern. "
                "Die Preset-ID bleibt unverändert."
            )
            with st.form("update_selected_configuration"):
                updated_name = st.text_input("Gespeicherter Name", value=selected_mode.name)
                updated_description = st.text_area(
                    "Gespeicherte Beschreibung", value=selected_mode.description, height=80
                )
                update_submitted = st.form_submit_button("Ausgewählte Konfiguration aktualisieren")
            if update_submitted:
                try:
                    updated_mode = preset_from_config(
                        config,
                        mode_id=selected_mode.mode_id,
                        name=updated_name.strip() or selected_mode.mode_id,
                        description=updated_description.strip(),
                    )
                    saved = save_custom_mode(updated_mode, paths=paths, overwrite=True)
                except (OSError, TypeError, ValueError) as exc:
                    st.error(str(exc))
                else:
                    queue_saved_preset(
                        saved,
                        f"Aktualisiert und geladen: {saved.name} ({saved.mode_id})",
                    )


st.set_page_config(page_title="Local Person ReID MVP", layout="wide")

paths = AppPaths()
paths.ensure()

st.title("Local Person Re-Identification MVP")
st.caption("Forschungsprototyp: YOLO, Tracking, qualitätsgefilterte ReID und lokale SQLite-Speicherung")
st.caption("B0/A1/A2/A3 und D1-D6 sind Ausgangspresets: D1 ist Details Tracking vollständig, D2-D6 entfernen jeweils eine Methode. Pilotwerte vor den Testclips einfrieren.")

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
    st.caption("Preset laden → Parameter bearbeiten → starten, als neues Preset speichern oder ein eigenes Preset aktualisieren. Nur die aktuellen Werte gelten beim Start.")
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
    if config.min_border_blur_score < config.min_initial_blur_score:
        st.info("Die Rand-Schärfe liegt unter der allgemeinen Initial-Schärfe. Effektiv gilt am Rand ebenfalls die höhere allgemeine Grenze.")
    st.caption("Schwellen auf Pilotdaten einstellen und vor der Evaluation einfrieren.")
    render_preset_forms(paths, config, selected_mode)
    with st.expander("Tatsächlich verwendete Pipeline-Konfiguration"):
        st.json(asdict(config))

    st.divider()
    st.header("Quelle und Anzeige (nicht Teil des Presets)")
    test_module = st.selectbox(
        "Test module",
        ["Mehrpersonen- und Trackingtest", "Einzelpersonen- und Detailtest"],
        help=(
            "Das Testmodul bestimmt nur Auswertung und Bericht. Die oben ausgewählte Pipeline "
            "bleibt unverändert: Beide Tests funktionieren mit jedem Preset."
        ),
    )
    if test_module == "Mehrpersonen- und Trackingtest":
        single_expected_person_id = ""
        single_condition = ""
        single_notes = ""
    else:
        st.caption("Kontrollierter Test mit genau einer realen Person; wertet die aktuell gewählte Pipeline mit Kontinuitäts- und Fragmentierungsmetriken aus.")
        single_expected_person_id = st.text_input("Expected person ID (optional)", value="")
        single_condition = st.text_input("Test condition", value="default")
        single_notes = st.text_area("Test notes", value="", height=70)
    input_type = st.radio("Input type", ["Video upload", "Local webcam"], index=0)
    isolated_run = st.checkbox("Isolierter Lauf (neue Datenbank)", value=True,
                               help="Standard für unabhängige Versuche. Deaktivieren verwendet den gemeinsamen Bestand dieses Encoders/Checkpoints. Andere Encoder haben eigene Datenbanken; für Registrierung/Rückkehr über zwei Videos den Versuchsstarter mit beiden Quellen verwenden.")
    shared_encoder_paths = paths_for_encoder(paths, config)
    if not isolated_run:
        st.caption(f"Gemeinsamer Profilbestand dieses Encoders: {shared_encoder_paths.db_path}")
    show_live_preview = st.checkbox("Show live annotated preview", value=True)
    preview_width = st.slider("Preview width", min_value=480, max_value=1400, value=960, step=40)

    st.divider()
    st.caption("Datenschutz-Hinweis: Das MVP nutzt synthetische IDs und sollte nur mit berechtigtem/consented Material getestet werden.")

source: str | int | CameraSource | None = None
uploaded_file = None
selected_camera_source: CameraSource | None = None
live_capture_seconds: float | None = None

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
    live_capture_seconds = float(st.number_input(
        "Live-Aufnahmedauer (Sekunden)",
        min_value=1,
        max_value=3600,
        value=30,
        step=5,
        help="Die Pipeline stoppt den Webcam-Lauf nach dieser real verstrichenen Zeit automatisch. Die Modellladezeit zählt nicht dazu.",
    ))

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
    displayed_paths = st.session_state.get("last_run_paths", shared_encoder_paths)
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
        run_paths = paths_for_encoder(run_paths, config)
        st.session_state["last_run_paths"] = run_paths
        pipeline = PersonReIdPipeline(config=config, paths=run_paths)
        result = pipeline.process(
            source,
            progress_callback=update_progress,
            frame_callback=update_live_preview,
            max_duration_seconds=live_capture_seconds if isinstance(source, CameraSource) else None,
        )
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
        st.metric("Decision policy", result.decision_policy)
        st.metric("Created persons in this run", result.created_persons)
        st.metric("Matched events in this run", result.matched_events)
        st.metric("Total known persons", len(result.persons))
        if result.decision_policy == "details_tracking_v2":
            st.metric("Strong matches", result.strong_match_events)
            st.metric("Pending weak matches", result.pending_weak_match_events)
            st.metric("Pending new-person observations", result.pending_new_person_events)

    if test_module == "Einzelpersonen- und Detailtest" and result.predictions_path:
        report = create_single_person_report(
            predictions_path=result.predictions_path,
            output_dir=result.predictions_path.parent,
            video_name=Path(str(source)).name,
            condition=single_condition,
            expected_person_id=single_expected_person_id.strip() or None,
            notes=single_notes,
        )
        st.subheader("Einzelpersonen-Metriken")
        metric_columns = st.columns(4)
        metric_columns[0].metric("Dominant Track Ratio", f"{report.metrics['dominant_track_ratio'] * 100:.1f} %")
        metric_columns[1].metric("Dominant Person Ratio", f"{report.metrics['dominant_person_ratio'] * 100:.1f} %")
        metric_columns[2].metric("Person-ID-Wechsel", report.metrics["person_switch_count"])
        metric_columns[3].metric("Fragmentierungsindex", report.metrics["profile_fragmentation_index"])
        st.caption(f"Einzelpersonen-Bericht: {report.json_path}")
        with st.expander("Vollständige Einzelpersonen-Metriken"):
            st.json(report.metrics)
    else:
        main_metrics = load_main_run_metrics(result.manifest_path)
        if main_metrics:
            st.subheader("Main-Laufzeit- und Artefaktmetriken")
            metric_columns = st.columns(3)
            metric_columns[0].metric("Processed frames", main_metrics.get("processed_frames") or 0)
            fps_value = main_metrics.get("frames_per_second")
            metric_columns[1].metric("Processing FPS", "n/a" if fps_value is None else f"{fps_value:.2f}")
            rtf_value = main_metrics.get("real_time_factor")
            metric_columns[2].metric("Real-time factor", "n/a" if rtf_value is None else f"{rtf_value:.2f}")
            with st.expander("Main-Artefakte"):
                st.json(main_metrics)

    if result.manifest_path and result.manifest_path.is_file():
        try:
            run_summary = json.loads(result.manifest_path.read_text(encoding="utf-8")).get("summary", {})
        except (OSError, json.JSONDecodeError):
            run_summary = {}
        if run_summary:
            with st.expander("Detailed run summary"):
                st.json(run_summary)

st.divider()

displayed_paths = st.session_state.get("last_run_paths", shared_encoder_paths)
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
