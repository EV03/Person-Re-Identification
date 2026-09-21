from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
from typing import Any

import cv2
import numpy as np

from app.utils.image_utils import normalize_vector
from app.evaluation.artifacts import file_reference, resolve_project_file, sha256_file
from app.pipeline.model_weights import DEFAULT_CHECKPOINT_PATH, DEFAULT_CHECKPOINT_SHA256, DEFAULT_CHECKPOINT_PROVENANCE


class ReIdEncoder(ABC):
    embedding_dim: int

    @abstractmethod
    def encode(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return a normalized embedding for one person crop in BGR format."""


class TorchreidOSNetEncoder(ReIdEncoder):
    """Optional OSNet encoder via torchreid.

    Install optional dependencies first:
        pip install -r requirements-optional-reid.txt

    If torchreid cannot load pretrained weights in your environment, install
    from the official repository:
        pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
    """

    def __init__(self, device: str = "cpu", model_name: str = "osnet_x1_0", *, checkpoint_path: str = "") -> None:
        if not checkpoint_path:
            raise ValueError("OSNet requires an explicit ReID checkpoint; no ImageNet/random fallback is allowed.")
        checkpoint = resolve_project_file(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"OSNet ReID checkpoint is missing: {checkpoint}. See docs/MODEL_WEIGHTS.md.")
        self.checkpoint_sha256 = sha256_file(checkpoint)
        if checkpoint.resolve() == resolve_project_file(DEFAULT_CHECKPOINT_PATH).resolve():
            if self.checkpoint_sha256 != DEFAULT_CHECKPOINT_SHA256:
                raise ValueError("The bundled-reference OSNet checkpoint checksum does not match the documented weights.")
        try:
            import torch
            try:
                module = importlib.import_module("torchreid.utils")
            except ModuleNotFoundError as exc:
                if exc.name != "torchreid.utils":
                    raise
                module = importlib.import_module("torchreid.reid.utils")
            FeatureExtractor = module.FeatureExtractor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                f"Torchreid backend could not be imported ({type(exc).__name__}: {exc}). "
                "Install requirements-optional-reid.txt or the documented evaluation lock."
            ) from exc

        if device == "auto":
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        # Use safe tensor-only loading, and reject incomplete/wrong backbone
        # checkpoints instead of accepting the library's partial-load warning.
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state)
        state = {key.removeprefix("module."): value for key, value in state.items()}
        self.extractor = FeatureExtractor(model_name=model_name, model_path=str(checkpoint),
                                          device=device, verbose=False)
        loaded = self.extractor.model.state_dict()
        required = [key for key in loaded if not key.startswith("classifier.") and not key.endswith("num_batches_tracked")]
        missing = [key for key in required if key not in state or state[key].shape != loaded[key].shape]
        if missing:
            raise ValueError(f"Checkpoint is incompatible with {model_name}; missing/mismatched backbone tensors: {missing[:5]}")
        if any(not torch.equal(loaded[key].cpu(), state[key].cpu()) for key in required):
            raise ValueError("OSNet did not load all required checkpoint tensors exactly.")
        self.embedding_dim = int(getattr(self.extractor.model, "feature_dim", 512))
        self.device = device
        self.model_name = model_name
        self.checkpoint_reference = file_reference(checkpoint)
        self.checkpoint_reference["provenance"] = (DEFAULT_CHECKPOINT_PROVENANCE
            if self.checkpoint_sha256 == DEFAULT_CHECKPOINT_SHA256
            else {"note": "User-supplied weights; document source and training data."})

    def describe_backend(self) -> dict[str, Any]:
        return {"encoder": {"backend": "torchreid", "model_name": self.model_name,
                            "checkpoint": self.checkpoint_reference}, "device": self.device}

    def encode(self, crop_bgr: np.ndarray) -> np.ndarray:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        features = self.extractor([crop_rgb])
        try:
            vector = features[0].detach().cpu().numpy().astype(np.float32)
        except AttributeError:
            vector = np.asarray(features[0], dtype=np.float32)
        if vector.size != self.embedding_dim or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError("OSNet returned an invalid, zero or incompatible embedding.")
        return normalize_vector(vector)


def build_encoder(device: str = "auto", *, model_name: str = "osnet_x1_0",
                  checkpoint_path: str = "") -> ReIdEncoder:
    """Build the project's single supported ReID encoder: OSNet via torchreid."""
    return TorchreidOSNetEncoder(device=device, model_name=model_name, checkpoint_path=checkpoint_path)
