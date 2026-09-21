"""Separate persistent profile databases by encoder, independently of matching."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version

from app.config import AppPaths, PipelineConfig
from app.evaluation.artifacts import resolve_project_file, sha256_file


def encoder_identity(config: PipelineConfig) -> dict[str, object]:
    """Ignore match/quality thresholds; those do not define the feature space."""
    identity: dict[str, object] = {"profile_format": "weighted-sum-v2", "backend": "torchreid"}
    checkpoint = resolve_project_file(config.reid_checkpoint)
    identity.update(model=config.reid_model_name,
                    checkpoint_sha256=sha256_file(checkpoint) if checkpoint.is_file() else None,
                    missing_checkpoint=str(checkpoint) if not checkpoint.is_file() else None,
                    preprocessing="torchreid-feature-extractor-defaults-v1")
    for package in ("torchreid", "torchvision"):
        try:
            identity[package] = version(package)
        except PackageNotFoundError:
            identity[package] = None
    return identity


def paths_for_encoder(paths: AppPaths, config: PipelineConfig) -> AppPaths:
    """Resolve a shared root or isolated experiment root; safe to call twice.

    Old unspecialized databases are left untouched. Different checkpoint files
    with identical contents share a namespace; changing weights creates a new one.
    Explicit custom/injected repositories remain the caller's responsibility.
    """
    identity = encoder_identity(config)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    root = paths.db_path.parent
    if root.parent.name == "encoders" and re.fullmatch(r"[0-9a-f]{64}", root.name):
        root = root.parent.parent
    return replace(paths, db_path=root / "encoders" / digest / paths.db_path.name)
