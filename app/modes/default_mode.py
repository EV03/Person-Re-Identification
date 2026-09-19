"""Four evaluation starting presets; calibrate and freeze copies on pilot data."""

from __future__ import annotations

from dataclasses import replace

from app.modes.base_mode import ModeConfig


def build_default_mode() -> ModeConfig:
    """B0: OSNet, quality gates and three initial observations."""
    return ModeConfig(
        mode_id="default",
        name="B0 - OSNet",
        description="Referenz: OSNet, qualitätsgefilterte Crops und drei Initialbeobachtungen.",
    )


def build_colorhist_mode() -> ModeConfig:
    """A1 differs from B0 only in the encoder."""
    return replace(
        build_default_mode(),
        mode_id="colorhist",
        name="A1 - Farbhistogramm",
        description="B0 mit Farbhistogramm statt OSNet; alle übrigen Parameter bleiben gleich.",
        encoder_backend="colorhist",
    )


def build_no_quality_thresholds_mode() -> ModeConfig:
    """A2 disables acceptance thresholds, not size checks or quality weights."""
    return replace(
        build_default_mode(),
        mode_id="no_quality_thresholds",
        name="A2 - Ohne Qualitätsschwellen",
        description="B0 mit beiden Qualitätsschwellen auf null; Mindestgrößen und Gewichtung bleiben erhalten.",
        min_embedding_quality=0.0,
        min_update_quality=0.0,
    )


def build_no_update_similarity_mode() -> ModeConfig:
    """A3 disables only the similarity gate; quality-filtered updates remain on."""
    return replace(
        build_default_mode(),
        mode_id="no_update_similarity",
        name="A3 - Ohne Update-Ähnlichkeitsschutz",
        description="B0 ohne Ähnlichkeitsprüfung vor Profilupdates; Qualitätsgrenzen und Updates bleiben erhalten.",
        min_update_similarity=-1.0,
    )


def build_details_tracking_mode() -> ModeConfig:
    """D1: OSNet plus the complete Details_Tracking decision policy."""
    return replace(
        build_default_mode(),
        mode_id="details_tracking",
        name="D1 - Details Tracking",
        description=(
            "OSNet plus Detail-Re-Ranking, Strong/Weak/Low-Zonen, verzögerte "
            "Neuanlage und begrenzten räumlichen Kontinuitätsbonus."
        ),
        decision_policy="details_tracking_v2",
    )


def build_details_no_reranking_mode() -> ModeConfig:
    """D2: D1 without detail extraction and detail-score re-ranking."""
    return replace(
        build_details_tracking_mode(),
        mode_id="details_no_reranking",
        name="D2 - Details ohne Detail-Re-Ranking",
        description="D1 ohne Detail-Registry und Detail-Re-Ranking; die übrigen Entscheidungsmechanismen bleiben aktiv.",
        detail_reranking_enabled=False,
    )


def build_details_no_weak_zone_mode() -> ModeConfig:
    """D3: D1 without tentative weak matches."""
    return replace(
        build_details_tracking_mode(),
        mode_id="details_no_weak_zone",
        name="D3 - Details ohne Weak-Zone",
        description="D1 ohne vorläufige Weak-Zuordnung; Scores unter der Strong-Schwelle gelten direkt als Low.",
        weak_match_zone_enabled=False,
    )


def build_details_immediate_new_person_mode() -> ModeConfig:
    """D4: D1 without accumulating low-score evidence before a new ID."""
    return replace(
        build_details_tracking_mode(),
        mode_id="details_immediate_new_person",
        name="D4 - Details ohne verzögerte Neuanlage",
        description="D1 legt nach dem Initialpuffer bei Low direkt eine neue ID an; der Überlappungsschutz bleibt aktiv.",
        delayed_new_person_enabled=False,
    )


def build_details_no_overlap_mode() -> ModeConfig:
    """D5: D1 without overlap protection for low-score candidates."""
    return replace(
        build_details_tracking_mode(),
        mode_id="details_no_overlap_protection",
        name="D5 - Details ohne Überlappungsschutz",
        description="D1 ohne Intersection-over-smaller-box-Schutz vor einer neuen ID.",
        overlap_protection_enabled=False,
    )


def build_details_no_motion_mode() -> ModeConfig:
    """D6: D1 without the image-space continuity bonus."""
    return replace(
        build_details_tracking_mode(),
        mode_id="details_no_motion_bonus",
        name="D6 - Details ohne Motion-Bonus",
        description="D1 ohne räumlichen Kontinuitätsbonus; visuelle und Detail-Scores bleiben unverändert aktiv.",
        motion_continuity_enabled=False,
    )
