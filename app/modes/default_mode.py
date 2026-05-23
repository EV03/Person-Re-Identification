from __future__ import annotations

from app.modes.base_mode import ModeConfig


def build_default_mode() -> ModeConfig:
    """Current project state as named default mode."""
    return ModeConfig(
        mode_id="default",
        name="Default ReID MVP",
        description=(
            "Bisheriger Stand: generische Personenerkennung, Tracking, "
            "Torchreid/OSNet-ReID, SQLite-Speicherung und Live-Preview."
        ),
        pipeline_type="person_reid",
        yolo_model="yolov8n.pt",
        tracker="bytetrack.yaml",
        encoder_backend="torchreid",
        match_threshold=0.82,
        detection_confidence=0.35,
        image_size=640,
        reid_every_n_frames=5,
        min_good_frames_before_reid=3,
        min_embedding_quality=0.55,
        min_update_quality=0.65,
        max_frames=500,
        device="auto",
        enable_motion_analysis=True,
        draw_motion_vectors=True,
        motion_max_jump_fraction=0.20,
        motion_smoothing_alpha=0.35,
        motion_min_displacement_px=2.0,
    )
