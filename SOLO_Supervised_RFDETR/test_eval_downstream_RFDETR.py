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

    def test_specimen_cluster_bootstrap_is_reproducible_and_keeps_observed_estimate_separate(self) -> None:
        schemes = {
            "two_class": (
                ["A", "A", "B", "B"],
                ["A", "B", "B", "A"],
                ["A", "B"],
            )
        }
        specimen_ids = ["Sample1", "Sample1", "Sample2", "Sample3"]
        first = downstream.bootstrap_accuracy_by_specimen(
            schemes, specimen_ids, replicates=500, seed=42
        )
        second = downstream.bootstrap_accuracy_by_specimen(
            schemes, specimen_ids, replicates=500, seed=42
        )
        self.assertEqual(first, second)
        ci = first["two_class"]["accuracy_ci_95"]
        self.assertLessEqual(ci["lower"], 0.5)
        self.assertGreaterEqual(ci["upper"], 0.5)
        self.assertEqual(first["two_class"]["clusters_resampled"], 3)
        self.assertEqual(first["two_class"]["valid_replicates"], 500)

    def test_minimum_count_discrepancy_is_distance_to_expert_bin(self) -> None:
        self.assertEqual(downstream.minimum_distance_from_expert_bin("26+", 25), 1)
        self.assertEqual(downstream.minimum_distance_from_expert_bin("26+", 24), 2)
        self.assertEqual(downstream.minimum_distance_from_expert_bin("0-9", 24), 15)
        self.assertEqual(downstream.minimum_distance_from_expert_bin("10-25", 17), 0)

    def test_secondary_minimum_discrepancy_covers_all_schemes(self) -> None:
        row = {
            "source_image_name": "Sample109 - BF.1_11.tif",
            "source_row": 40,
            "epithelial_count_bin": "26+",
            "leucocyte_count_bin": "0-9",
            "n_squamous_epithelial_cell": 25,
            "n_leucocyte": 0,
            "geckler_class": "G1",
            "predicted_geckler_class": "G6",
            "collapsed_geckler_label": "Unacceptable",
            "predicted_collapsed_geckler_label": "Unknown",
            "murray_washington_label": "Unacceptable",
            "predicted_murray_washington_label": "Unacceptable",
        }
        summary, detail_rows = downstream.minimum_count_discrepancy_analysis([row])
        self.assertEqual(summary["schemes"]["geckler"]["misclassified_fov_count"], 1)
        self.assertEqual(
            summary["schemes"]["collapsed_geckler"]["sum_across_cell_types"][
                "mean_minimum_discrepancy"
            ],
            1,
        )
        self.assertEqual(summary["schemes"]["murray_washington"]["misclassified_fov_count"], 0)
        self.assertEqual(len(detail_rows), 2)
        self.assertEqual(detail_rows[0]["minimum_epithelial_count_discrepancy"], 1)

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
