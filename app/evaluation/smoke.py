"""Real model startup/embedding smoke test, not a ReID accuracy measurement."""

import argparse
import numpy as np

from app.config import PipelineConfig
from app.pipeline.reid_encoder import build_encoder


def main() -> None:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=defaults.reid_checkpoint)
    parser.add_argument("--model", default=defaults.reid_model_name)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    encoder = build_encoder("torchreid", device=args.device, model_name=args.model,
                            checkpoint_path=args.checkpoint)
    crop = np.full((256, 128, 3), 128, dtype=np.uint8)
    first, second = encoder.encode(crop), encoder.encode(crop)
    if not np.allclose(first, second) or not np.isclose(np.linalg.norm(first), 1):
        raise RuntimeError("Embedding is not deterministic/normalized in this environment.")
    print(f"OSNet OK: {first.shape}, norm={np.linalg.norm(first):.6f}, sha256={encoder.checkpoint_sha256}")


if __name__ == "__main__":
    main()
