from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.orchestrator import open_output_video_writer


class VideoOutputTests(unittest.TestCase):
    def test_output_writer_creates_readable_mp4(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output.mp4"
            writer, codec = open_output_video_writer(output_path, 25.0, (64, 96))
            try:
                for value in range(12):
                    frame = np.full((96, 64, 3), value * 10, dtype=np.uint8)
                    writer.write(frame)
            finally:
                writer.release()

            capture = cv2.VideoCapture(str(output_path))
            try:
                self.assertTrue(capture.isOpened())
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 12)
                ok, frame = capture.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape[:2], (96, 64))
            finally:
                capture.release()

            self.assertEqual(codec, "h264")


if __name__ == "__main__":
    unittest.main()
