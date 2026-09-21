"""Three OSNet evaluation starting presets; calibrate and freeze copies on pilot data."""

from __future__ import annotations

from dataclasses import replace

from app.modes.base_mode import ModeConfig


def build_default_mode() -> ModeConfig:
    """B0: OSNet, quality gates and five spaced initial observations."""
    return ModeConfig(
        mode_id="default",
        name="B0 - OSNet",
        description="Referenz: OSNet, qualitätsgefilterte Crops und fünf zeitlich getrennte Initialbeobachtungen.",
    )


def build_no_quality_thresholds_mode() -> ModeConfig:
    """A2 disables acceptance thresholds, not size checks or quality weights."""
    return replace(
        build_default_mode(),
        mode_id="no_quality_thresholds",
        name="A2 - Ohne Qualitätsschwellen",
        description="B0 mit allen Qualitätsschwellen auf null; Mindestgrößen und Gewichtung bleiben erhalten.",
        min_embedding_quality=0.0,
        min_initial_blur_score=0.0,
        min_border_blur_score=0.0,
        min_initial_aspect_ratio_score=0.0,
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
