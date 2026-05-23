from __future__ import annotations

from app.modes.base_mode import ModeConfig


def build_football_mode() -> ModeConfig:
    """Initial football analysis mode.

    The first version deliberately reuses the stable ReID pipeline and only
    activates football-specific configuration flags. Ball detection, pitch
    mapping, team classification and statistics aggregation are provided as
    placeholders and can be wired in one by one.
    """
    return ModeConfig(
        mode_id="football_team_analysis",
        name="Football Team Analysis",
        description=(
            "Fußballmodus für Videoeingaben. V1 nutzt die stabile ReID-Pipeline "
            "und bereitet Balltracking, Teamklassifikation, Spielfeld-Mapping "
            "und Statistik-Aggregation vor."
        ),
        pipeline_type="football_analysis",
        yolo_model="yolov8n.pt",
        tracker="botsort.yaml",
        encoder_backend="torchreid",
        match_threshold=0.84,
        detection_confidence=0.30,
        image_size=640,
        reid_every_n_frames=5,
        max_frames=1000,
        device="auto",
        enable_ball_tracking=True,
        enable_pitch_mapping=True,
        enable_team_classification=True,
        enable_stats_aggregation=True,
    )
