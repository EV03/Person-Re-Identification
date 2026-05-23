from __future__ import annotations

from dataclasses import dataclass
import sys

import cv2


@dataclass(frozen=True)
class CameraSource:
    """A local OpenCV camera source.

    Streamlit itself runs on the local Python process. Therefore this source refers
    to a camera index visible to OpenCV on that machine, not to the browser's
    MediaDevices API.
    """

    index: int
    backend_name: str = "auto"

    def label(self) -> str:
        return f"Camera {self.index} [{self.backend_name}]"


@dataclass(frozen=True)
class CameraInfo:
    index: int
    backend_name: str
    backend_id: int
    width: int
    height: int
    fps: float

    @property
    def key(self) -> str:
        return f"{self.backend_name}:{self.index}"

    @property
    def label(self) -> str:
        resolution = f"{self.width}x{self.height}" if self.width and self.height else "unknown resolution"
        fps = f", {self.fps:.0f} FPS" if self.fps and self.fps > 1 else ""
        return f"Camera {self.index} [{self.backend_name}] - {resolution}{fps}"


def camera_backends() -> list[tuple[str, int]]:
    """Return OpenCV camera backends worth trying on the current platform."""
    if sys.platform.startswith("win"):
        return [
            ("dshow", cv2.CAP_DSHOW),
            ("msmf", cv2.CAP_MSMF),
            ("auto", cv2.CAP_ANY),
        ]
    return [("auto", cv2.CAP_ANY)]


def backend_id_from_name(name: str) -> int:
    normalized = (name or "auto").lower().strip()
    for backend_name, backend_id in camera_backends():
        if backend_name == normalized:
            return backend_id
    return cv2.CAP_ANY


def open_camera_capture(source: CameraSource) -> cv2.VideoCapture:
    backend_id = backend_id_from_name(source.backend_name)
    if backend_id == cv2.CAP_ANY:
        return cv2.VideoCapture(int(source.index))
    return cv2.VideoCapture(int(source.index), backend_id)


def probe_camera(index: int, backend_name: str = "auto") -> CameraInfo | None:
    source = CameraSource(index=int(index), backend_name=backend_name)
    cap = open_camera_capture(source)
    try:
        if not cap.isOpened():
            return None

        ok, frame = cap.read()
        if not ok or frame is None:
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(frame.shape[1])
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(frame.shape[0])
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return CameraInfo(
            index=int(index),
            backend_name=backend_name,
            backend_id=backend_id_from_name(backend_name),
            width=width,
            height=height,
            fps=fps,
        )
    finally:
        cap.release()


def scan_local_cameras(max_index: int = 5, backend_name: str = "dshow") -> list[CameraInfo]:
    """Scan local OpenCV camera indexes for one backend.

    OpenCV usually exposes numeric indexes instead of friendly camera names. The
    returned metadata makes it easier to choose the right webcam.
    """
    cameras: list[CameraInfo] = []
    for index in range(max(0, int(max_index)) + 1):
        camera = probe_camera(index=index, backend_name=backend_name)
        if camera is not None:
            cameras.append(camera)
    return cameras


def read_single_preview_frame(source: CameraSource) -> tuple[bool, object | None]:
    cap = open_camera_capture(source)
    try:
        if not cap.isOpened():
            return False, None
        ok, frame = cap.read()
        return bool(ok and frame is not None), frame
    finally:
        cap.release()
