from __future__ import annotations

import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from app.config import PROJECT_ROOT as _configured_project_root  # noqa: F401

    import cv2
    import imageio_ffmpeg
    import numpy
    import pandas
    import qdrant_client
    import streamlit
    import torch
    import torchreid
    import torchvision
    import ultralytics

    from app.utils.model_discovery import discover_osnet_models, discover_yolo_models

    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Torch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"Torchreid: {getattr(torchreid, '__version__', 'installed')}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"ImageIO-FFmpeg: {imageio_ffmpeg.__version__} ({imageio_ffmpeg.get_ffmpeg_exe()})")
    print(f"NumPy: {numpy.__version__}")
    print(f"Pandas: {pandas.__version__}")
    print(f"Streamlit: {streamlit.__version__}")
    print(f"Qdrant client: {getattr(qdrant_client, '__version__', 'installed')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    yolo_models = discover_yolo_models()
    osnet_models = discover_osnet_models()
    print("Local YOLO models:")
    for model in yolo_models:
        print(f"- {model.display_name}: {model.path}")
    print("Local OSNet models:")
    for model in osnet_models:
        print(f"- {model.display_name}: {model.path}")

    if platform.python_version() != "3.10.8":
        raise RuntimeError("Dieses Projekt erwartet exakt Python 3.10.8.")
    if not yolo_models:
        raise RuntimeError("Kein lokales YOLO-Modell gefunden.")
    if not osnet_models:
        raise RuntimeError("Kein lokales OSNet-Modell gefunden.")


if __name__ == "__main__":
    main()
