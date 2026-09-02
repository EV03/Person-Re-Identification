from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.utils.upload_utils import persist_uploaded_video, sanitize_uploaded_filename


class UploadUtilsTests(unittest.TestCase):
    def test_sanitizes_path_components_and_special_characters(self) -> None:
        self.assertEqual(sanitize_uploaded_filename("../../team video?.MP4"), ("team_video", ".mp4"))
        self.assertEqual(sanitize_uploaded_filename(r"C:\temp\match.mov"), ("match", ".mov"))

    def test_rejects_unsupported_extensions(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_uploaded_filename("not-a-video.exe")

    def test_same_upload_is_reused_across_reruns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            first = persist_uploaded_video(input_dir, "match.mp4", b"same video")
            second = persist_uploaded_video(input_dir, "match.mp4", b"same video")

            self.assertEqual(first, second)
            self.assertEqual(list(input_dir.iterdir()), [first])
            self.assertEqual(first.read_bytes(), b"same video")

    def test_different_content_gets_a_different_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            first = persist_uploaded_video(input_dir, "match.mp4", b"version one")
            second = persist_uploaded_video(input_dir, "match.mp4", b"version two")

            self.assertNotEqual(first, second)
            self.assertEqual(len(list(input_dir.iterdir())), 2)

    def test_size_limit_is_enforced_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            with self.assertRaises(ValueError):
                persist_uploaded_video(input_dir, "match.mp4", b"1234", max_bytes=3)

            self.assertFalse(input_dir.exists())


if __name__ == "__main__":
    unittest.main()
