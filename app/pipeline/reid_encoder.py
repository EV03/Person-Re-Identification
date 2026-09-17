from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

from app.utils.image_utils import normalize_vector
from app.utils.model_discovery import find_osnet_model


class ReIdEncoder(ABC):
    embedding_dim: int

    @abstractmethod
    def encode(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return a normalized embedding for one person crop in BGR format."""


class TorchreidOSNetEncoder(ReIdEncoder):
    """OSNet encoder via the project's required Torchreid dependency."""

    _EMBEDDING_DIMENSIONS = {
        "osnet_x1_0": 512,
        "osnet_ibn_x1_0": 512,
        "osnet_ain_x1_0": 512,
        "osnet_x0_75": 384,
        "osnet_x0_5": 256,
        "osnet_x0_25": 128,
    }

    def __init__(self, device: str = "cpu", model_name: str = "osnet_x1_0", model_path: str = "") -> None:
        try:
            from torchreid.utils import FeatureExtractor
        except Exception as exc:
            raise RuntimeError(
                "Torchreid/OSNet ist nicht installiert. Führe scripts/setup_windows.ps1 aus."
            ) from exc

        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        resolved_model_path = Path(model_path).expanduser().resolve() if model_path else None
        if resolved_model_path is None or not resolved_model_path.is_file():
            discovered = find_osnet_model(model_name)
            resolved_model_path = discovered.path if discovered is not None else None
        if resolved_model_path is None or not resolved_model_path.is_file():
            raise RuntimeError(
                f"Kein lokales Gewicht für '{model_name}' gefunden. "
                "Installiere es mit scripts/install_models_windows.ps1."
            )

        self.model_name = model_name
        self.model_path = resolved_model_path
        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=str(resolved_model_path),
            device=device,
        )
        self.embedding_dim = self._EMBEDDING_DIMENSIONS.get(model_name, 512)

    def encode(self, crop_bgr: np.ndarray) -> np.ndarray:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        features = self.extractor([crop_rgb])
        try:
            vector = features[0].detach().cpu().numpy().astype(np.float32)
        except AttributeError:
            vector = np.asarray(features[0], dtype=np.float32)
        return normalize_vector(vector)


def build_encoder(model_name: str, model_path: str = "", device: str = "auto") -> ReIdEncoder:
    return TorchreidOSNetEncoder(device=device, model_name=model_name, model_path=model_path)
