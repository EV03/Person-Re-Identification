from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
runtime_dir = PROJECT_ROOT / ".runtime"
runtime_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_dir))
YOLO_DIR = PROJECT_ROOT / "models" / "yolo"
OSNET_DIR = PROJECT_ROOT / "models" / "reid"

SUPPORTED_YOLO_MODELS = {
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
}
SUPPORTED_OSNET_MODELS = {
    "osnet_x1_0",
    "osnet_x0_75",
    "osnet_x0_5",
    "osnet_x0_25",
    "osnet_ibn_x1_0",
}


def _install_yolo(model_name: str) -> Path:
    if model_name not in SUPPORTED_YOLO_MODELS:
        supported = ", ".join(sorted(SUPPORTED_YOLO_MODELS))
        raise ValueError(f"Nicht unterstütztes YOLO-Downloadziel '{model_name}'. Erlaubt: {supported}")

    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    target = YOLO_DIR / model_name
    if target.exists():
        print(f"YOLO bereits installiert: {target}")
        return target

    existing = PROJECT_ROOT / model_name
    if existing.exists():
        shutil.copy2(existing, target)
        print(f"YOLO aus Projektroot übernommen: {target}")
        return target

    from ultralytics import YOLO

    previous_directory = Path.cwd()
    try:
        os.chdir(YOLO_DIR)
        YOLO(model_name)
    finally:
        os.chdir(previous_directory)

    if not target.exists():
        raise RuntimeError(f"Ultralytics hat das erwartete Modell nicht unter {target} abgelegt.")
    print(f"YOLO heruntergeladen: {target}")
    return target


def _install_osnet(model_name: str) -> Path:
    if model_name not in SUPPORTED_OSNET_MODELS:
        supported = ", ".join(sorted(SUPPORTED_OSNET_MODELS))
        raise ValueError(f"Nicht unterstütztes OSNet-Downloadziel '{model_name}'. Erlaubt: {supported}")

    OSNET_DIR.mkdir(parents=True, exist_ok=True)
    target = OSNET_DIR / f"{model_name}_imagenet.pth"
    if target.exists():
        print(f"OSNet bereits installiert: {target}")
        return target

    cached = Path.home() / ".cache" / "torch" / "checkpoints" / target.name
    if cached.exists():
        shutil.copy2(cached, target)
        print(f"OSNet aus lokalem Torch-Cache übernommen: {target}")
        return target

    import gdown
    from torchreid.models.osnet import pretrained_urls

    url = pretrained_urls.get(model_name)
    if not url:
        raise RuntimeError(f"Torchreid stellt für {model_name} keine bekannte Download-URL bereit.")
    downloaded = gdown.download(url, str(target), quiet=False)
    if not downloaded or not target.exists():
        raise RuntimeError(f"OSNet konnte nicht nach {target} heruntergeladen werden.")
    print(f"OSNet heruntergeladen: {target}")
    return target


def _list_models() -> None:
    from app.utils.model_discovery import discover_osnet_models, discover_yolo_models

    print("Lokale YOLO-Modelle:")
    for model in discover_yolo_models():
        print(f"- {model.display_name}: {model.path}")
    print("Lokale OSNet-Modelle:")
    for model in discover_osnet_models():
        print(f"- {model.display_name}: {model.path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Projektlokale YOLO- und OSNet-Modelle verwalten.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    install = subparsers.add_parser("install", help="Bekannte Modelle installieren oder aus lokalem Cache übernehmen.")
    install.add_argument("--yolo", action="append", default=[], help="YOLO-Modellname, mehrfach verwendbar.")
    install.add_argument("--osnet", action="append", default=[], help="OSNet-Modellname, mehrfach verwendbar.")
    subparsers.add_parser("list", help="Lokal erkannte Modelle anzeigen.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "list":
        _list_models()
        return
    if not args.yolo and not args.osnet:
        raise ValueError("Mindestens ein YOLO- oder OSNet-Modell angeben.")
    for model_name in args.yolo:
        _install_yolo(model_name)
    for model_name in args.osnet:
        _install_osnet(model_name)
    _list_models()


if __name__ == "__main__":
    main()
