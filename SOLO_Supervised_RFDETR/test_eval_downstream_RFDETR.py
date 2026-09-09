from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval_downstream_RFDETR as downstream
import qa_inference_RFDETR as qa


class DownstreamEvaluationTests(unittest.TestCase):
    def test_locked_inference_constants(self) -> None:
        settings = qa.downstream_inference_settings()
        self.assertEqual(settings["slice_height"], 640)
        self.assertEqual(settings["slice_width"], 640)
        self.assertEqual(settings["class_score_thresholds"], downstream.EXPECTED_THRESHOLDS)
        self.assertEqual(settings["postprocess"]["type"], "GREEDYNMM")
        self.assertEqual(settings["postprocess"]["match_metric"], "IOU")
        self.assertEqual(settings["postprocess"]["match_threshold"], 0.50)
        self.assertFalse(settings["postprocess"]["class_agnostic"])

    def test_clinical_label_parser(self) -> None:
        self.assertEqual(downstream.normalized_label("Qualified"), 1)
        self.assertEqual(downstream.normalized_label("Partially Qualified"), 2)
        self.assertEqual(downstream.normalized_label("Not Qualified"), 3)
        with self.assertRaises(ValueError):
            downstream.normalized_label("unknown")

    def test_expert_count_bins_and_published_schemes(self) -> None:
        self.assertEqual(downstream.normalized_count_bin("10 til 25"), "10-25")
        self.assertEqual(downstream.normalized_count_bin("10"), "10-25")
        self.assertEqual(downstream.normalized_count_bin("25"), "10-25")
        self.assertEqual(downstream.normalized_count_bin("26"), "26+")
        self.assertEqual(downstream.geckler_class("26+", "0-9"), "G1")
        self.assertEqual(downstream.geckler_class("26+", "10-25"), "G2")
        self.assertEqual(downstream.geckler_class("26+", "26+"), "G3")
        self.assertEqual(downstream.geckler_class("10-25", "26+"), "G4")
        self.assertEqual(downstream.geckler_class("0-9", "26+"), "G5")
        self.assertEqual(downstream.geckler_class("0-9", "0-9"), "G6")
        self.assertEqual(downstream.collapsed_geckler_label("G4"), "Acceptable")
        self.assertEqual(downstream.collapsed_geckler_label("G5"), "Acceptable")
        self.assertEqual(downstream.collapsed_geckler_label("G6"), "Unknown")
        self.assertEqual(
            downstream.murray_washington_label("0-9", "26+"), "Acceptable"
        )
        self.assertEqual(
            downstream.murray_washington_label("10-25", "26+"), "Unacceptable"
        )

    def test_quality_rule_boundaries(self) -> None:
        self.assertEqual(qa.classify_quality_from_counts(9, 0)[0], 3)
        self.assertEqual(qa.classify_quality_from_counts(10, 0)[0], 2)
        self.assertEqual(qa.classify_quality_from_counts(26, 0)[0], 1)
        self.assertEqual(qa.classify_quality_from_counts(51, 9)[0], 1)
        self.assertEqual(qa.classify_quality_from_counts(51, 10)[0], 2)
        self.assertEqual(qa.classify_quality_from_counts(51, 26)[0], 2)

    def test_metric_calculation_and_confusion_orientation(self) -> None:
        true_ids = [1, 1, 2, 2, 3, 3]
        predicted_ids = [1, 2, 2, 3, 3, 1]
        metrics, per_class, matrix = downstream.calculate_metrics(
            true_ids,
            predicted_ids,
        )
        self.assertEqual(matrix, [[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 0.5)
        self.assertAlmostEqual(metrics["cohen_kappa"], 0.25)
        self.assertAlmostEqual(metrics["mean_absolute_class_error"], 4 / 6)
        self.assertAlmostEqual(metrics["within_one_class_accuracy"], 5 / 6)
        self.assertTrue(all(row["support"] == 2 for row in per_class))

    def test_image_resolution_never_enters_patch_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sample_dir = root / "Sample 1"
            patches_dir = sample_dir / "Patches for Sample 1"
            patches_dir.mkdir(parents=True)
            full_fov = sample_dir / "Sample1 - BF.1_1.tif"
            full_fov.touch()
            (patches_dir / full_fov.name).touch()

            record = downstream.ClinicalRecord(
                source_row=2,
                source_image_name=full_fov.name,
                manual_label_id=1,
                manual_label="Qualified",
                epithelial_count_bin="0-9",
                leucocyte_count_bin="26+",
                geckler_class="G5",
                collapsed_geckler_label="Acceptable",
                murray_washington_label="Acceptable",
                annotator="",
                comment="",
            )
            resolved = downstream.resolve_clinical_images([record], root)
            self.assertEqual(Path(resolved[0].image_path), full_fov.resolve())


if __name__ == "__main__":
    unittest.main()
