from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from app.utils.image_utils import normalize_vector


class ReIdEncoder(ABC):
    embedding_dim: int

    @abstractmethod
    def encode(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return a normalized embedding for one person crop in BGR format."""


class ColorHistogramEncoder(ReIdEncoder):
    """Very small demo encoder based on HSV color histograms.

    This is intentionally simple and fast. It is useful for proving the complete
    pipeline without installing heavy ReID dependencies. It is not robust enough
    for production-grade person re-identification.
    """

    def __init__(self, bins_h: int = 16, bins_s: int = 8, bins_v: int = 8) -> None:
        self.bins_h = bins_h
        self.bins_s = bins_s
        self.bins_v = bins_v
        self.embedding_dim = bins_h + bins_s + bins_v

    def encode(self, crop_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop_bgr, (128, 256), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        hist_h = cv2.calcHist([hsv], [0], None, [self.bins_h], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [self.bins_s], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [self.bins_v], [0, 256]).flatten()

        vector = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
        return normalize_vector(vector)


class TorchreidOSNetEncoder(ReIdEncoder):
    """Optional OSNet encoder via torchreid.

    Install optional dependencies first:
        pip install -r requirements-optional-reid.txt

    If torchreid cannot load pretrained weights in your environment, install
    from the official repository:
        pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
    """

    def __init__(self, device: str = "cpu", model_name: str = "osnet_x1_0") -> None:
        try:
            from torchreid.utils import FeatureExtractor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Torchreid backend requested, but torchreid is not installed. "
                "Install requirements-optional-reid.txt first."
            ) from exc

        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        self.extractor = FeatureExtractor(model_name=model_name, model_path="", device=device)
        self.embedding_dim = 512

    def encode(self, crop_bgr: np.ndarray) -> np.ndarray:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        features = self.extractor([crop_rgb])
        try:
            vector = features[0].detach().cpu().numpy().astype(np.float32)
        except AttributeError:
            vector = np.asarray(features[0], dtype=np.float32)
        return normalize_vector(vector)


def build_encoder(backend: str, device: str = "auto") -> ReIdEncoder:
    backend = backend.lower().strip()
    if backend == "colorhist":
        return ColorHistogramEncoder()
    if backend == "torchreid":
        return TorchreidOSNetEncoder(device=device)
    raise ValueError(f"Unknown encoder backend: {backend}")
