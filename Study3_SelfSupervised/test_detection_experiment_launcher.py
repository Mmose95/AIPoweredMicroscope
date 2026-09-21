from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_detection_experiments import _materialize_dataset


class DetectionExperimentDatasetTest(unittest.TestCase):
    def test_budget_is_train_only_and_test_is_not_materialized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            images = root / "images"
            output = root / "effective"
            categories = [{"id": 0, "name": "Cell", "supercategory": "none"}]
            for split, count in (("train", 4), ("valid", 2), ("test", 1)):
                split_images = []
                annotations = []
                for index in range(count):
                    relative = Path(f"Sample_{index:03d}") / f"{split}_{index}.jpg"
                    image_path = images / relative
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8), color=(index, 0, 0)).save(image_path)
                    split_images.append(
                        {"id": index, "file_name": relative.as_posix(), "width": 8, "height": 8}
                    )
                    annotations.append(
                        {
                            "id": index,
                            "image_id": index,
                            "category_id": 0,
                            "bbox": [1, 1, 2, 2],
                            "area": 4,
                            "iscrowd": 0,
                        }
                    )
                split_dir = source / split
                split_dir.mkdir(parents=True)
                (split_dir / "_annotations.coco.json").write_text(
                    json.dumps(
                        {"images": split_images, "annotations": annotations, "categories": categories}
                    ),
                    encoding="utf-8",
                )

            report = _materialize_dataset(source, output, images, budget=0.5, seed=7)
            train = json.loads((output / "train" / "_annotations.coco.json").read_text())
            valid = json.loads((output / "valid" / "_annotations.coco.json").read_text())

            self.assertEqual(len(train["images"]), 2)
            self.assertEqual(len(train["annotations"]), 2)
            self.assertEqual(len(valid["images"]), 2)
            self.assertTrue(all(Path(item["file_name"]).is_absolute() for item in train["images"]))
            self.assertFalse((output / "test").exists())
            self.assertFalse(report["test_materialized"])


if __name__ == "__main__":
    unittest.main()
