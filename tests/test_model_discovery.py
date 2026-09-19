from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.utils.model_discovery import (
    discover_reid_models,
    discover_yolo_models,
    display_labels,
    reid_architectures,
    selectable_paths,
)


class ModelDiscoveryTests(unittest.TestCase):
    def test_discovers_project_models_and_keeps_custom_values(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            yolo = root / "models" / "yolo" / "yolov8m.pt"
            reid = root / "models" / "reid" / "osnet_x0_5_market1501.pth"
            ignored = root / "models" / "reid" / "notes.txt"
            for path in (yolo, reid, ignored):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")

            yolo_models = discover_yolo_models(root)
            reid_models = discover_reid_models(root)
            self.assertEqual([model.path for model in yolo_models], [yolo.resolve()])
            self.assertIn(reid.resolve(), [model.path for model in reid_models])
            self.assertIn("osnet_x0_5", reid_architectures(reid_models))
            self.assertEqual(selectable_paths("custom/model.pt", yolo_models, root),
                             ["custom/model.pt", "models/yolo/yolov8m.pt"])
            self.assertIn("models/reid/osnet_x0_5_market1501.pth", display_labels(reid_models, root))


if __name__ == "__main__":
    unittest.main()
