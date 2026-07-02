from __future__ import annotations

import os

from app.modes.base_mode import ModeConfig


def build_max_accuracy_mode() -> ModeConfig:
    """Resource-heavy preset for maximum person detection and ReID quality."""
    return ModeConfig(
        mode_id="max_accuracy_qdrant_botsort",
        name="Max Accuracy ReID",
        description=(
            "Ressourcenintensiver Modus: größtes stabiles YOLOv8-Modell, "
            "BoT-SORT Tracking, Torchreid/OSNet Embeddings und Qdrant als Vector Store. Standardmäßig nutzt dieser Modus Qdrant Local ohne Docker. "
            "Die eigene Bewegungsrichtungsanalyse ist deaktiviert, weil BoT-SORT das Tracking übernimmt."
        ),
        pipeline_type="person_reid",
        yolo_model=os.getenv("REID_MAX_YOLO_MODEL", "yolov8x.pt"),
        tracker="botsort.yaml",
        encoder_backend="torchreid",
        vector_store_backend="qdrant",
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "person_reid_max_accuracy"),
        qdrant_mode=os.getenv("QDRANT_MODE", "local"),
        qdrant_local_path=os.getenv("QDRANT_LOCAL_PATH", "data/qdrant_local_max_accuracy"),
        qdrant_prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "false").lower() in {"1", "true", "yes", "on"},
        match_threshold=0.86,
        detection_confidence=0.25,
        image_size=1280,
        reid_every_n_frames=1,
        min_good_frames_before_reid=5,
        min_embedding_quality=0.60,
        min_update_quality=0.75,
        max_frames=100000,
        min_crop_height=96,
        min_crop_width=36,
        crop_padding=0.08,
        device="auto",
        draw_debug=True,
        live_preview_every_n_frames=10,
        enable_motion_analysis=False,
        draw_motion_vectors=False,
        disable_internal_motion_when_botsort=True,
        motion_max_jump_fraction=0.20,
        motion_smoothing_alpha=0.35,
        motion_min_displacement_px=2.0,
        enable_detail_analysis=True,
        detail_weight=0.15,
        detail_min_confidence=0.55,
        draw_detail_labels=True,
    )
