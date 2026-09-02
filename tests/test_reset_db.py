from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.reset_db import safe_reset_target


class SafeResetTargetTests(unittest.TestCase):
    def test_accepts_child_of_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data"
            target = root / "output"

            self.assertEqual(safe_reset_target(target, root), target.resolve())

    def test_rejects_allowed_root_itself(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data"

            with self.assertRaises(ValueError):
                safe_reset_target(root, root)

    def test_rejects_parent_and_sibling_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            root = base / "data"

            for unsafe_target in (base, base / "other", root / ".."):
                with self.subTest(target=unsafe_target):
                    with self.assertRaises(ValueError):
                        safe_reset_target(unsafe_target, root)


if __name__ == "__main__":
    unittest.main()
