#!/usr/bin/env python
"""Build one Word report covering detector, calibration, and downstream results.

The report is generated from saved evaluation artifacts. No model inference or
threshold optimization is performed here. Microsoft Word is automated through
the locally installed COM interface so no additional document package is
required in the ai_powmic environment.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)

try:
    import win32com.client
except Exception as exc:  # pragma: no cover - Windows dependency gate
    raise ImportError(
        "pywin32 and a local Microsoft Word installation are required."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OBJECT_OUTPUT = (
    PROJECT_ROOT
    / "EvaluationOutput"
    / "TwoClass_HPO009_EMA_TestEval_20260701-140821"
)
CALIBRATION_OUTPUT = (
    PROJECT_ROOT
    / "EvaluationOutput"
    / "RFDETR_ThresholdCalibration_20260701-140419"
)
DOWNSTREAM_ROOT = PROJECT_ROOT / "DownstreamOutput_Downstream"
SPLIT_SUMMARY = (
    SCRIPT_DIR
    / "Stat_Dataset"
    / "QA-2025v1_TwoClass_OVR_V2_20260618-101346"
    / "split_summary.json"
)
TEST_COCO = SPLIT_SUMMARY.parent / "test" / "_annotations.coco.json"
HPO_RECORD = Path(
    r"E:\PHD\Results\Quality Assessment\FINAL_B200"
    r"\session_20260618_113853\TwoClass\HPO_Config_009\hpo_record.json"
)
TRAIN_KWARGS = HPO_RECORD.parent / "run_meta" / "train_kwargs.json"
CODE_COMMIT = "27c95e8875da3f11b486a03f522a0a27e82ab529"
CODE_URL = (
    "https://github.com/Mmose95/AIPoweredMicroscope/commit/"
    + CODE_COMMIT
)

WD_ALIGN_LEFT = 0
WD_ALIGN_CENTER = 1
WD_BREAK_PAGE = 7
WD_AUTOFIT_CONTENT = 1
WD_FORMAT_DOCUMENT_DEFAULT = 16
WD_PAPER_A4 = 7
WD_STATISTIC_PAGES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-output", type=Path, default=OBJECT_OUTPUT)
    parser.add_argument("--calibration-output", type=Path, default=CALIBRATION_OUTPUT)
    parser.add_argument("--downstream-output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def latest_downstream_output(root: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and (path / "downstream_evaluation_summary.json").is_file()
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No completed downstream output was found in {root}")
    return candidates[-1]


def rgb(red: int, green: int, blue: int) -> int:
    """Convert RGB components to the integer representation used by Word."""
    return int(red) + int(green) * 256 + int(blue) * 65536


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def pct(value: Any, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}%"


def threshold_text(thresholds: dict[str, Any]) -> str:
    return "; ".join(
        f"{class_name}={float(value):.2f}"
        for class_name, value in thresholds.items()
    )


def sample_id(image_name: str) -> str:
    match = re.search(r"Sample\s*(\d+)", image_name, re.IGNORECASE)
    return f"Sample{match.group(1)}" if match else image_name


def cluster_bootstrap(
    rows: Sequence[dict[str, str]],
    replicates: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]], int]:
    true_ids = np.asarray([int(row["manual_label_id"]) for row in rows])
    predicted_ids = np.asarray([int(row["predicted_label_id"]) for row in rows])
    clusters: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        clusters[sample_id(row["source_image_name"])].append(index)
    cluster_names = sorted(clusters)

    def metric_vector(y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
        return [
            float(accuracy_score(y_true, y_pred)),
            float(balanced_accuracy_score(y_true, y_pred)),
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=[1, 2, 3],
                    average="macro",
                    zero_division=0,
                )
            ),
            float(cohen_kappa_score(y_true, y_pred)),
            float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        ]

    rng = np.random.default_rng(seed)
    bootstrapped = np.empty((replicates, 5), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for replicate in range(replicates):
            selected = rng.choice(
                cluster_names,
                size=len(cluster_names),
                replace=True,
            )
            indices = np.concatenate([clusters[name] for name in selected])
            bootstrapped[replicate] = metric_vector(
                true_ids[indices],
                predicted_ids[indices],
            )

    estimates = metric_vector(true_ids, predicted_ids)
    names = [
        "Accuracy",
        "Balanced accuracy",
        "Macro F1",
        "Cohen's kappa",
        "Quadratic-weighted kappa",
    ]
    intervals = {
        name: {
            "estimate": estimates[index],
            "low": float(np.quantile(bootstrapped[:, index], 0.025)),
            "high": float(np.quantile(bootstrapped[:, index], 0.975)),
        }
        for index, name in enumerate(names)
    }
    return intervals, len(cluster_names)


def per_class_standard_coco(
    test_coco_path: Path,
    predictions_path: Path,
) -> list[dict[str, Any]]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(test_coco_path))
        detections = coco.loadRes(str(predictions_path))
        rows: list[dict[str, Any]] = []
        for category in coco.loadCats(coco.getCatIds()):
            evaluator = COCOeval(coco, detections, "bbox")
            evaluator.params.catIds = [category["id"]]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            rows.append(
                {
                    "class": category["name"],
                    "AP@50:95": float(evaluator.stats[0]),
                    "AP@50": float(evaluator.stats[1]),
                    "AP@75": float(evaluator.stats[2]),
                    "AR@100": float(evaluator.stats[8]),
                }
            )
    return rows


def downstream_descriptive_analysis(
    rows: Sequence[dict[str, str]],
) -> dict[str, Any]:
    true_ids = np.asarray([int(row["manual_label_id"]) for row in rows])
    predicted_ids = np.asarray([int(row["predicted_label_id"]) for row in rows])
    difference = predicted_ids - true_ids

    count_summary: dict[str, dict[str, Any]] = {}
    for label_id, label_name in (
        (1, "Qualified"),
        (2, "Partially Qualified"),
        (3, "Not Qualified"),
    ):
        selected = [row for row in rows if int(row["manual_label_id"]) == label_id]
        leucocytes = np.asarray([int(row["n_leucocyte"]) for row in selected])
        epithelial = np.asarray(
            [int(row["n_squamous_epithelial_cell"]) for row in selected]
        )
        count_summary[label_name] = {
            "n": len(selected),
            "leucocyte_median": float(np.median(leucocytes)),
            "leucocyte_q1": float(np.quantile(leucocytes, 0.25)),
            "leucocyte_q3": float(np.quantile(leucocytes, 0.75)),
            "epithelial_median": float(np.median(epithelial)),
            "epithelial_q1": float(np.quantile(epithelial, 0.25)),
            "epithelial_q3": float(np.quantile(epithelial, 0.75)),
        }

    misclassified = [
        {
            "image": row["source_image_name"],
            "clinical": row["manual_label"],
            "predicted": row["predicted_label"],
            "leucocyte": int(row["n_leucocyte"]),
            "epithelial": int(row["n_squamous_epithelial_cell"]),
            "quality_score": int(row["total_quality_score"]),
        }
        for row in rows
        if row["manual_label_id"] != row["predicted_label_id"]
    ]

    return {
        "error_direction": {
            "exact": int(np.sum(difference == 0)),
            "predicted_more_severe": int(np.sum(difference > 0)),
            "predicted_more_favorable": int(np.sum(difference < 0)),
            "two_class_error": int(np.sum(np.abs(difference) == 2)),
        },
        "aggregate_detections": {
            "raw": sum(int(row["n_predictions_raw"]) for row in rows),
            "kept_before_cross_class": sum(
                int(row["n_predictions_kept_before_duplicate_suppression"])
                for row in rows
            ),
            "cross_class_suppressed": sum(
                int(row["n_cross_class_duplicates_suppressed"]) for row in rows
            ),
            "final": sum(int(row["n_predictions_kept"]) for row in rows),
            "leucocyte": sum(int(row["n_leucocyte"]) for row in rows),
            "epithelial": sum(
                int(row["n_squamous_epithelial_cell"]) for row in rows
            ),
        },
        "counts_by_clinical_class": count_summary,
        "misclassified": misclassified,
    }


class WordReport:
    def __init__(self) -> None:
        self.word = win32com.client.DispatchEx("Word.Application")
        self.word.Visible = False
        self.word.DisplayAlerts = 0
        self.document = self.word.Documents.Add()
        self.selection = self.word.Selection
        self._configure_document()

    def _configure_document(self) -> None:
        for section in self.document.Sections:
            section.PageSetup.PaperSize = WD_PAPER_A4
            section.PageSetup.TopMargin = 2.0 * 28.3464567
            section.PageSetup.BottomMargin = 1.8 * 28.3464567
            section.PageSetup.LeftMargin = 2.0 * 28.3464567
            section.PageSetup.RightMargin = 2.0 * 28.3464567

        styles = self.document.Styles
        normal = styles.Item("Normal")
        normal.Font.Name = "Aptos"
        normal.Font.Size = 10
        normal.ParagraphFormat.SpaceAfter = 6
        normal.ParagraphFormat.LineSpacing = 13

        style_settings = {
            "Title": (24, (31, 78, 121)),
            "Subtitle": (13, (75, 85, 99)),
            "Heading 1": (16, (31, 78, 121)),
            "Heading 2": (13, (46, 116, 181)),
            "Heading 3": (11, (55, 65, 81)),
            "Caption": (9, (75, 85, 99)),
        }
        for name, (size, color) in style_settings.items():
            style = styles.Item(name)
            style.Font.Name = "Aptos"
            style.Font.Size = size
            style.Font.Color = rgb(*color)
            if name.startswith("Heading"):
                style.Font.Bold = True
                style.ParagraphFormat.KeepWithNext = True

        for section in self.document.Sections:
            footer = section.Footers.Item(1)
            footer.Range.Text = (
                "RF-DETR microscopy quality assessment — combined results  |  "
            )
            footer.Range.Font.Name = "Aptos"
            footer.Range.Font.Size = 8
            footer.Range.Font.Color = rgb(100, 116, 139)
            footer.Range.ParagraphFormat.Alignment = WD_ALIGN_CENTER
            footer.PageNumbers.Add()

        try:
            self.document.BuiltInDocumentProperties("Title").Value = (
                "RF-DETR Object Detection, Calibration, and Downstream Results"
            )
            self.document.BuiltInDocumentProperties("Subject").Value = (
                "End-to-end evaluation of microscopy specimen quality assessment"
            )
        except Exception:
            pass

    def paragraph(
        self,
        text: str = "",
        style: str = "Normal",
        alignment: int = WD_ALIGN_LEFT,
        bold: bool = False,
        italic: bool = False,
    ) -> None:
        self.selection.Style = self.document.Styles.Item(style)
        self.selection.ParagraphFormat.Alignment = alignment
        self.selection.Font.Bold = bold
        self.selection.Font.Italic = italic
        self.selection.TypeText(str(text))
        self.selection.TypeParagraph()
        self.selection.Font.Bold = False
        self.selection.Font.Italic = False
        self.selection.ParagraphFormat.Alignment = WD_ALIGN_LEFT

    def bullets(self, items: Iterable[str]) -> None:
        for item in items:
            self.paragraph(str(item), style="List Bullet")

    def page_break(self) -> None:
        self.selection.InsertBreak(WD_BREAK_PAGE)

    def table(
        self,
        headers: Sequence[str],
        rows: Sequence[Sequence[Any]],
        font_size: float = 9,
    ) -> None:
        table = self.document.Tables.Add(
            self.selection.Range,
            len(rows) + 1,
            len(headers),
        )
        table.Style = "Table Grid"
        table.AllowAutoFit = True
        table.AutoFitBehavior(WD_AUTOFIT_CONTENT)
        table.Rows.Item(1).HeadingFormat = True
        table.Rows.Item(1).Range.Font.Bold = True
        table.Rows.Item(1).Range.Font.Color = rgb(255, 255, 255)
        table.Rows.Item(1).Shading.BackgroundPatternColor = rgb(31, 78, 121)

        for column, value in enumerate(headers, start=1):
            table.Cell(1, column).Range.Text = str(value)
        for row_index, values in enumerate(rows, start=2):
            table.Rows.Item(row_index).AllowBreakAcrossPages = False
            if row_index % 2 == 1:
                table.Rows.Item(row_index).Shading.BackgroundPatternColor = rgb(
                    238,
                    244,
                    250,
                )
            for column, value in enumerate(values, start=1):
                table.Cell(row_index, column).Range.Text = str(value)

        table.Range.Font.Name = "Aptos"
        table.Range.Font.Size = font_size
        self.selection.SetRange(table.Range.End, table.Range.End)
        self.selection.TypeParagraph()

    def image(
        self,
        path: Path,
        caption: str,
        width_inches: float = 6.25,
    ) -> None:
        if not path.is_file():
            self.paragraph(f"[Figure unavailable: {path}]", italic=True)
            return
        self.selection.Style = self.document.Styles.Item("Normal")
        self.selection.ParagraphFormat.Alignment = WD_ALIGN_CENTER
        shape = self.selection.InlineShapes.AddPicture(
            str(path.resolve()),
            False,
            True,
        )
        maximum_width = 72.0 * width_inches
        if shape.Width > maximum_width:
            shape.LockAspectRatio = True
            shape.Width = maximum_width
        self.selection.SetRange(shape.Range.End, shape.Range.End)
        self.selection.TypeParagraph()
        self.paragraph(caption, style="Caption", alignment=WD_ALIGN_CENTER)

    def add_contents(self) -> None:
        self.paragraph("Contents", style="Heading 1")
        table_of_contents = self.document.TablesOfContents.Add(
            Range=self.selection.Range,
            UseHeadingStyles=True,
            UpperHeadingLevel=1,
            LowerHeadingLevel=3,
            UseHyperlinks=True,
        )
        self.selection.SetRange(
            table_of_contents.Range.End,
            table_of_contents.Range.End,
        )
        self.selection.TypeParagraph()
        self.page_break()

    def save(self, path: Path) -> dict[str, int]:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.document.Fields.Update()
        for toc in self.document.TablesOfContents:
            toc.Update()
        self.document.Repaginate()
        self.document.SaveAs2(
            str(path.resolve()),
            FileFormat=WD_FORMAT_DOCUMENT_DEFAULT,
        )
        statistics = {
            "pages": int(self.document.ComputeStatistics(WD_STATISTIC_PAGES)),
            "paragraphs": int(self.document.Paragraphs.Count),
            "tables": int(self.document.Tables.Count),
            "figures": int(self.document.InlineShapes.Count),
        }
        self.document.Close(SaveChanges=False)
        self.word.Quit()
        return statistics

    def close_without_saving(self) -> None:
        try:
            self.document.Close(SaveChanges=False)
        finally:
            self.word.Quit()


def build_report(args: argparse.Namespace) -> tuple[Path, dict[str, int]]:
    object_output = args.object_output.resolve()
    calibration_output = args.calibration_output.resolve()
    downstream_output = (
        args.downstream_output.resolve()
        if args.downstream_output
        else latest_downstream_output(DOWNSTREAM_ROOT).resolve()
    )
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = (
            PROJECT_ROOT
            / "EvaluationOutput"
            / f"Combined_RFDETR_Study_Report_{datetime.now():%Y%m%d-%H%M%S}"
        )
    report_path = output_dir / "RFDETR_Combined_Study_Results.docx"

    object_summary = read_json(object_output / "eval_summary.json")
    object_confusion = read_json(object_output / "confusion_matrix.json")
    calibration = read_json(calibration_output / "threshold_calibration_summary.json")
    downstream = read_json(downstream_output / "downstream_evaluation_summary.json")
    split = read_json(SPLIT_SUMMARY)
    hpo = read_json(HPO_RECORD)
    train_kwargs = read_json(TRAIN_KWARGS)
    downstream_rows = read_csv(downstream_output / "downstream_predictions.csv")
    per_class_coco = per_class_standard_coco(
        TEST_COCO,
        object_output / "predictions_coco.json",
    )
    bootstrap, n_clusters = cluster_bootstrap(
        downstream_rows,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    descriptive = downstream_descriptive_analysis(downstream_rows)

    report = WordReport()
    try:
        report.paragraph(
            "End-to-End Evaluation of RF-DETR-Based\n"
            "Microscopy Specimen Quality Assessment",
            style="Title",
            alignment=WD_ALIGN_CENTER,
        )
        report.paragraph(
            "Object detection • independent threshold calibration • "
            "final expert-referenced downstream evaluation",
            style="Subtitle",
            alignment=WD_ALIGN_CENTER,
        )
        report.paragraph(
            f"Generated {datetime.now():%d %B %Y}",
            alignment=WD_ALIGN_CENTER,
        )
        report.paragraph("")
        report.table(
            ["Item", "Locked value"],
            [
                ["Detector", "RFDETR2XLarge, HPO configuration 009, EMA checkpoint"],
                ["Internal model resolution", "880 × 880 pixels"],
                ["SAHI full-FOV slice size", "640 × 640 pixels with 20% overlap"],
                [
                    "Class confidence thresholds",
                    threshold_text(calibration["selected_thresholds"]),
                ],
                ["Final downstream set", "98 FOVs from 34 samples"],
                ["Code provenance", f"{CODE_COMMIT[:8]} — {CODE_URL}"],
            ],
        )
        report.paragraph("Headline outcomes", style="Heading 2")
        report.table(
            ["Evaluation stage", "Primary result"],
            [
                [
                    "Held-out object detection",
                    "COCO AP@50:95 "
                    + fmt(object_summary["coco_standard"]["AP@50:95"], 4)
                    + "; AP@50 "
                    + fmt(object_summary["coco_standard"]["AP@50"], 4),
                ],
                [
                    "Independent calibration",
                    "Macro F1 "
                    + fmt(calibration["selected_joint_metrics"]["macro_f1"], 4)
                    + " at Leucocyte=0.36 and Epithelial=0.35",
                ],
                [
                    "Final downstream task",
                    "Accuracy "
                    + pct(downstream["metrics"]["accuracy"], 2)
                    + "; balanced accuracy "
                    + pct(downstream["metrics"]["balanced_accuracy"], 2)
                    + "; macro F1 "
                    + fmt(downstream["metrics"]["macro_f1"], 4),
                ],
            ],
        )
        report.page_break()
        report.add_contents()

        report.paragraph("1. Executive summary", style="Heading 1")
        report.paragraph(
            "This report consolidates the complete three-stage evaluation of the "
            "final two-class RF-DETR model. The detector was first assessed on a "
            "held-out object-detection test split. Class-specific confidence "
            "thresholds were then selected on a separate calibration set. Those "
            "thresholds and all downstream inference settings were locked before "
            "the expert-referenced downstream FOV labels were analyzed."
        )
        report.bullets(
            [
                "The detector achieved COCO AP@50:95 of "
                + fmt(object_summary["coco_standard"]["AP@50:95"], 4)
                + " and AP@50 of "
                + fmt(object_summary["coco_standard"]["AP@50"], 4)
                + " on 331 held-out 640 × 640 patches containing 4,004 objects.",
                "Independent calibration on 53 patches containing 776 objects "
                "selected confidence thresholds of 0.36 for Leucocyte and 0.35 "
                "for Squamous Epithelial Cell; joint macro F1 was "
                + fmt(calibration["selected_joint_metrics"]["macro_f1"], 4)
                + ".",
                "The final downstream evaluation included 98 expert-labeled FOVs "
                "after four prespecified exclusions. Accuracy was "
                + pct(downstream["metrics"]["accuracy"], 2)
                + ", balanced accuracy was "
                + pct(downstream["metrics"]["balanced_accuracy"], 2)
                + ", and macro F1 was "
                + fmt(downstream["metrics"]["macro_f1"], 4)
                + ".",
                "Downstream errors were predominantly conservative: "
                f"{descriptive['error_direction']['predicted_more_severe']} of "
                f"{len(descriptive['misclassified'])} errors assigned a worse "
                "quality category than the clinical reference, while "
                f"{descriptive['error_direction']['predicted_more_favorable']} "
                "assigned a more favorable category.",
                "No threshold or quality-rule parameter was changed after the "
                "downstream labels were inspected.",
            ]
        )

        report.paragraph("2. Evaluation design and data separation", style="Heading 1")
        report.paragraph(
            "The study design deliberately separates model fitting, standard "
            "detector testing, threshold calibration, and final downstream "
            "testing. This prevents downstream test performance from influencing "
            "the confidence thresholds used to decide whether detections enter "
            "the count-based quality rule."
        )
        counts = split["counts"]
        report.table(
            ["Partition/stage", "Samples", "Images", "Leucocyte", "Epithelial", "Purpose"],
            [
                [
                    "Detector training",
                    counts["train"]["n_samples"],
                    counts["train"]["n_images"],
                    counts["train"]["n_target_boxes_by_class"]["Leucocyte"],
                    counts["train"]["n_target_boxes_by_class"][
                        "Squamous Epithelial Cell"
                    ],
                    "Model fitting",
                ],
                [
                    "Detector validation",
                    counts["valid"]["n_samples"],
                    counts["valid"]["n_images"],
                    counts["valid"]["n_target_boxes_by_class"]["Leucocyte"],
                    counts["valid"]["n_target_boxes_by_class"][
                        "Squamous Epithelial Cell"
                    ],
                    "HPO/model selection",
                ],
                [
                    "Detector test",
                    counts["test"]["n_samples"],
                    counts["test"]["n_images"],
                    counts["test"]["n_target_boxes_by_class"]["Leucocyte"],
                    counts["test"]["n_target_boxes_by_class"][
                        "Squamous Epithelial Cell"
                    ],
                    "Held-out object detection",
                ],
                [
                    "Calibration",
                    calibration["task_count"],
                    calibration["image_count"],
                    calibration["annotation_counts"]["Leucocyte"],
                    calibration["annotation_counts"]["Squamous Epithelial Cell"],
                    "Threshold selection only",
                ],
                [
                    "Downstream test",
                    n_clusters,
                    downstream["dataset"]["included_image_count"],
                    "—",
                    "—",
                    "Final FOV quality evaluation",
                ],
            ],
        )
        report.paragraph(
            "The detector split was sample-wise (28/9/11 samples for "
            "train/validation/test). The downstream workbook contained 102 "
            "non-header image rows: 98 were eligible, one image was marked as "
            "not present in the annotation project, and three were considered "
            "too difficult for a reliable clinical classification."
        )

        report.paragraph("3. Final model and locked inference settings", style="Heading 1")
        report.paragraph("3.1 Model selection and training", style="Heading 2")
        report.table(
            ["Parameter", "Value"],
            [
                ["Selected configuration", "HPO configuration 009"],
                ["Architecture", hpo["MODEL_CLS"]],
                ["Checkpoint", "checkpoint_best_ema.pth"],
                ["Internal resolution", hpo["RESOLUTION"]],
                ["Input data mode", hpo["input_mode"] + " pixel source patches"],
                ["Queries", hpo["NUM_QUERIES"]],
                ["Batch size / gradient accumulation", f"{hpo['BATCH']} / {hpo['GRAD_ACCUM_STEPS']}"],
                ["Learning rate", f"{hpo['LR']:.1e}"],
                ["Weight decay / dropout", f"{hpo['WEIGHT_DECAY']} / {hpo['DROPOUT']}"],
                ["Multi-scale training", str(hpo["MULTI_SCALE"])],
                ["Maximum epochs / best epoch", f"{hpo['EPOCHS']} / {hpo['best_epoch']}"],
                ["Validation AP@50", fmt(hpo["val_AP50"], 4)],
                ["Validation mAP@50:95", fmt(hpo["val_mAP5095"], 4)],
                ["Random seed", hpo["SEED"]],
                ["Patch size / attention windows", f"{train_kwargs['patch_size']} / {train_kwargs['num_windows']}"],
            ],
        )
        report.paragraph(
            "The microscopy patches remained 640 × 640 at the data interface. "
            "RF-DETR internally resized them to the selected 880 × 880 model "
            "resolution. The same distinction was retained for downstream SAHI: "
            "the slice is 640 × 640, while the model internally operates at 880."
        )

        report.paragraph("3.2 Locked downstream inference", style="Heading 2")
        settings = downstream["preflight"]["inference_settings"]
        cross_class = settings["cross_class_duplicate_suppression"]
        report.table(
            ["Setting", "Locked value"],
            [
                ["SAHI slice", f"{settings['slice_width']} × {settings['slice_height']}"],
                [
                    "Slice overlap",
                    f"{pct(settings['overlap_width_ratio'], 0)} horizontal and vertical",
                ],
                ["Standard whole-image prediction", str(settings["perform_standard_prediction"])],
                [
                    "Class confidence thresholds",
                    threshold_text(settings["class_score_thresholds"]),
                ],
                [
                    "Within-class postprocess",
                    f"{settings['postprocess']['type']}, "
                    f"{settings['postprocess']['match_metric']}="
                    f"{settings['postprocess']['match_threshold']:.2f}, "
                    "class-aware",
                ],
                [
                    "Cross-class duplicate suppression",
                    f"IOS≥{cross_class['ios_threshold']:.2f}, "
                    f"IoU≥{cross_class['iou_threshold']:.2f}, "
                    f"area ratio≥{cross_class['area_ratio_threshold']:.2f}",
                ],
            ],
        )

        report.paragraph("3.3 Count-to-quality rule", style="Heading 2")
        report.table(
            ["Component", "Observed count", "Score"],
            [
                ["Leucocyte", "<10", "−1"],
                ["Leucocyte", "10–25", "0"],
                ["Leucocyte", "26–50", "+1"],
                ["Leucocyte", ">50", "+2"],
                ["Squamous epithelial", "<10", "0"],
                ["Squamous epithelial", "10–25", "−1"],
                ["Squamous epithelial", ">25", "−2"],
            ],
        )
        report.paragraph(
            "The component scores were summed. An FOV was Qualified when the "
            "total score was at least +1 and fewer than 10 epithelial cells were "
            "detected; it was Not Qualified when the total was −1 or lower; all "
            "remaining cases were Partially Qualified."
        )

        report.paragraph("4. Held-out object-detection evaluation", style="Heading 1")
        report.paragraph(
            "Standard COCO ranking metrics were calculated from detections "
            f"retained at the numerical score floor of {object_summary['score_floor']}. "
            "They do not use the calibrated thresholds. The confusion matrix and "
            "operating-point precision/recall/F1 are threshold-dependent and use "
            "the independently calibrated class-specific thresholds at IoU=0.50."
        )
        overall = object_summary["coco_standard"]
        report.table(
            ["Metric", "Overall result"],
            [
                ["COCO AP@50:95", fmt(overall["AP@50:95"], 4)],
                ["AP@50", fmt(overall["AP@50"], 4)],
                ["AP@75", fmt(overall["AP@75"], 4)],
                ["AR@1", fmt(overall["AR@1"], 4)],
                ["AR@10", fmt(overall["AR@10"], 4)],
                ["AR@100", fmt(overall["AR@100"], 4)],
            ],
        )
        report.paragraph("Per-class standard COCO metrics", style="Heading 2")
        report.table(
            ["Class", "AP@50:95", "AP@50", "AP@75", "AR@100"],
            [
                [
                    row["class"],
                    fmt(row["AP@50:95"], 4),
                    fmt(row["AP@50"], 4),
                    fmt(row["AP@75"], 4),
                    fmt(row["AR@100"], 4),
                ]
                for row in per_class_coco
            ],
        )
        operating = object_summary["operating_point"]
        report.paragraph("Locked operating point at IoU=0.50", style="Heading 2")
        report.table(
            ["Thresholds", "TP", "FP", "FN", "Precision", "Recall", "F1", "FP/image"],
            [
                [
                    threshold_text(operating["class_score_thresholds"]),
                    operating["tp"],
                    operating["fp"],
                    operating["fn"],
                    fmt(operating["precision"], 4),
                    fmt(operating["recall"], 4),
                    fmt(operating["f1"], 4),
                    fmt(operating["fp_per_image"], 3),
                ]
            ],
        )
        report.image(
            object_output / "confusion_matrix.png",
            "Figure 1. Object-detection confusion matrix at the locked "
            "class-specific confidence thresholds and IoU=0.50. Rows are "
            "ground truth; columns are predictions. Background indicates misses "
            "or unmatched detections.",
        )
        report.image(
            object_output / "map_by_iou_threshold.png",
            "Figure 2. Detector AP across IoU thresholds. The standard headline "
            "COCO result remains AP@50:95.",
        )
        object_matrix = object_confusion["matrix"]
        report.paragraph(
            "The test set contained 2,851 annotated leucocytes and 1,153 "
            "squamous epithelial cells. At the locked operating point, the "
            f"confusion matrix recorded {object_matrix[0][0]:,} correctly "
            f"localized leucocytes and {object_matrix[1][1]:,} correctly "
            "localized epithelial cells. Epithelial cells showed higher "
            "localization performance than leucocytes across AP@50:95, AP@50, "
            "AP@75, and AR@100."
        )
        report.paragraph(
            "The saved test threshold sweep is descriptive sensitivity analysis "
            "only. Its test-set maximum was not used to select or revise the "
            "operating thresholds and is intentionally not reported as a "
            "selected operating point."
        )

        report.paragraph("5. Independent confidence-threshold calibration", style="Heading 1")
        report.paragraph(
            "Calibration used 53 images from 11 tasks, with 572 leucocyte and "
            "204 squamous epithelial annotations. Predictions were matched at "
            "IoU=0.50. Thresholds from 0.00 to 0.95 were evaluated at 0.01 "
            "increments, with extra observed-score candidates, and macro F1 was "
            "the prespecified selection criterion."
        )
        selected = calibration["selected_thresholds"]
        per_class = calibration["best_per_class_by_f1"]
        report.table(
            ["Class", "Threshold", "TP", "FP", "FN", "Precision", "Recall", "F1", "Jaccard"],
            [
                [
                    class_name,
                    fmt(selected[class_name], 2),
                    per_class[class_name]["tp"],
                    per_class[class_name]["fp"],
                    per_class[class_name]["fn"],
                    fmt(per_class[class_name]["precision"], 4),
                    fmt(per_class[class_name]["recall"], 4),
                    fmt(per_class[class_name]["f1"], 4),
                    fmt(per_class[class_name]["jaccard"], 4),
                ]
                for class_name in ("Leucocyte", "Squamous Epithelial Cell")
            ],
        )
        joint = calibration["selected_joint_metrics"]
        report.table(
            ["Joint metric", "Result"],
            [
                ["TP / FP / FN", f"{joint['tp']} / {joint['fp']} / {joint['fn']}"],
                ["Micro precision", fmt(joint["micro_precision"], 4)],
                ["Micro recall", fmt(joint["micro_recall"], 4)],
                ["Micro F1", fmt(joint["micro_f1"], 4)],
                ["Macro precision", fmt(joint["macro_precision"], 4)],
                ["Macro recall", fmt(joint["macro_recall"], 4)],
                ["Macro F1", fmt(joint["macro_f1"], 4)],
                ["Macro Jaccard", fmt(joint["macro_jaccard"], 4)],
            ],
        )
        report.image(
            calibration_output / "leucocyte_threshold_sweep.png",
            "Figure 3. Leucocyte precision, recall, F1, and Jaccard across "
            "confidence thresholds. The selected F1-maximizing threshold was 0.36.",
        )
        report.image(
            calibration_output / "squamous_epithelial_cell_threshold_sweep.png",
            "Figure 4. Squamous epithelial cell calibration curves. The selected "
            "F1-maximizing threshold was 0.35.",
        )
        report.image(
            calibration_output / "joint_macro_f1_heatmap.png",
            "Figure 5. Joint macro-F1 surface for the two class-specific "
            "confidence thresholds.",
        )
        report.paragraph(
            "The calibrated thresholds were subsequently treated as fixed "
            "method parameters. They were applied to the threshold-dependent "
            "detector results and carried unchanged into downstream SAHI "
            "inference."
        )

        report.paragraph("6. Final downstream quality evaluation", style="Heading 1")
        report.paragraph("6.1 Cohort flow and runtime", style="Heading 2")
        report.table(
            ["Item", "Count/result"],
            [
                ["Workbook rows reviewed", downstream["preflight"]["workbook_rows"]],
                ["Eligible FOVs analyzed", downstream["dataset"]["included_image_count"]],
                ["Represented samples", n_clusters],
                ["Excluded FOVs", downstream["dataset"]["excluded_image_count"]],
                ["Qualified references", downstream["dataset"]["manual_label_counts"]["Qualified"]],
                ["Partially Qualified references", downstream["dataset"]["manual_label_counts"]["Partially Qualified"]],
                ["Not Qualified references", downstream["dataset"]["manual_label_counts"]["Not Qualified"]],
                ["Total evaluation time", f"{downstream['runtime']['total_evaluation_seconds']:.1f} s"],
                ["Mean detector time per FOV", f"{downstream['runtime']['mean_inference_seconds_per_image']:.2f} s"],
            ],
        )
        report.paragraph(
            "All 98 eligible images completed without inference failure, and an "
            "annotated overlay was produced for every included FOV."
        )

        report.paragraph("6.2 Overall downstream performance", style="Heading 2")
        report.paragraph(
            f"Uncertainty intervals are percentile 95% confidence intervals from "
            f"{args.bootstrap_replicates:,} cluster-bootstrap replicates, "
            f"resampling the {n_clusters} source samples with replacement "
            f"(seed {args.bootstrap_seed}). This preserves within-sample FOV "
            "correlation better than an image-level bootstrap."
        )
        report.table(
            ["Metric", "Estimate", "Cluster-bootstrap 95% CI"],
            [
                [
                    metric,
                    fmt(values["estimate"], 4),
                    f"{fmt(values['low'], 4)}–{fmt(values['high'], 4)}",
                ]
                for metric, values in bootstrap.items()
            ],
        )
        report.table(
            ["Additional metric", "Result"],
            [
                ["Weighted precision", fmt(downstream["metrics"]["weighted_precision"], 4)],
                ["Weighted recall", fmt(downstream["metrics"]["weighted_recall"], 4)],
                ["Weighted F1", fmt(downstream["metrics"]["weighted_f1"], 4)],
                ["Mean absolute class error", fmt(downstream["metrics"]["mean_absolute_class_error"], 4)],
                ["Within-one-class accuracy", fmt(downstream["metrics"]["within_one_class_accuracy"], 4)],
            ],
        )

        report.paragraph("6.3 Class-level performance", style="Heading 2")
        report.table(
            ["Class", "Support", "TP", "FP", "FN", "Precision", "Recall", "F1", "Specificity"],
            [
                [
                    row["label"],
                    row["support"],
                    row["tp"],
                    row["fp"],
                    row["fn"],
                    fmt(row["precision"], 4),
                    fmt(row["recall"], 4),
                    fmt(row["f1"], 4),
                    fmt(row["specificity"], 4),
                ]
                for row in downstream["per_class_metrics"]
            ],
        )
        report.image(
            downstream_output / "downstream_confusion_matrix.png",
            "Figure 6. Final downstream confusion matrix. Rows are clinical "
            "reference labels and columns are locked pipeline predictions.",
        )
        matrix = downstream["confusion_matrix"]["counts"]
        report.paragraph(
            f"The pipeline correctly classified {sum(matrix[index][index] for index in range(3))} "
            "of 98 FOVs. Qualified predictions had precision 1.00: no clinically "
            "Partially Qualified or Not Qualified image was predicted as "
            "Qualified. However, Qualified recall was "
            f"{downstream['per_class_metrics'][0]['recall']:.3f}, reflecting a "
            "conservative acceptance threshold. Partially Qualified was the "
            "most difficult class (F1 "
            f"{downstream['per_class_metrics'][1]['f1']:.3f}), and Not Qualified "
            "recall was "
            f"{downstream['per_class_metrics'][2]['recall']:.3f}."
        )

        report.paragraph("6.4 Error direction and detected-count patterns", style="Heading 2")
        errors = descriptive["error_direction"]
        report.table(
            ["Outcome", "FOVs", "Percentage of all FOVs"],
            [
                ["Exact class agreement", errors["exact"], pct(errors["exact"] / len(downstream_rows), 1)],
                [
                    "Predicted more severe than reference",
                    errors["predicted_more_severe"],
                    pct(errors["predicted_more_severe"] / len(downstream_rows), 1),
                ],
                [
                    "Predicted more favorable than reference",
                    errors["predicted_more_favorable"],
                    pct(errors["predicted_more_favorable"] / len(downstream_rows), 1),
                ],
                [
                    "Two-category errors",
                    errors["two_class_error"],
                    pct(errors["two_class_error"] / len(downstream_rows), 1),
                ],
            ],
        )
        report.table(
            ["Clinical class", "n", "Leucocyte median (IQR)", "Epithelial median (IQR)"],
            [
                [
                    class_name,
                    values["n"],
                    f"{fmt(values['leucocyte_median'], 1)} "
                    f"({fmt(values['leucocyte_q1'], 1)}–{fmt(values['leucocyte_q3'], 1)})",
                    f"{fmt(values['epithelial_median'], 1)} "
                    f"({fmt(values['epithelial_q1'], 1)}–{fmt(values['epithelial_q3'], 1)})",
                ]
                for class_name, values in descriptive[
                    "counts_by_clinical_class"
                ].items()
            ],
        )
        aggregate = descriptive["aggregate_detections"]
        report.paragraph(
            f"Across all FOVs, {aggregate['raw']:,} raw sliced predictions were "
            f"reduced to {aggregate['kept_before_cross_class']:,} candidates by "
            "the locked confidence thresholds and within-class SAHI postprocess. "
            f"Cross-class suppression removed {aggregate['cross_class_suppressed']:,} "
            f"additional duplicates, leaving {aggregate['final']:,} detections "
            f"({aggregate['leucocyte']:,} leucocytes and "
            f"{aggregate['epithelial']:,} epithelial cells)."
        )

        report.paragraph("7. Integrated interpretation", style="Heading 1")
        report.bullets(
            [
                "The final RFDETR2XLarge detector showed strong held-out "
                "localization performance, especially for squamous epithelial "
                "cells. Leucocyte detection remained the weaker component.",
                "Independent calibration produced high object-level F1 and "
                "provided a defensible, prespecified operating point for any "
                "threshold-dependent result.",
                "The lower downstream performance does not invalidate the "
                "detector or calibration results. It measures the entire chain: "
                "sliced full-FOV inference, duplicate handling, class-specific "
                "thresholding, count aggregation, and the deterministic clinical "
                "quality rule.",
                "Most downstream errors were conservative. This explains why "
                "Qualified precision was perfect while Qualified recall was "
                "substantially lower.",
                "The Partially Qualified category showed the greatest overlap "
                "with the deterministic rule boundaries and was most frequently "
                "assigned to Not Qualified.",
                "The final downstream set must remain untouched for optimization. "
                "Any future revision of thresholds or the count-to-quality rule "
                "should be developed on a new training/development cohort and "
                "then tested on an independent external set.",
            ]
        )

        report.paragraph("8. Limitations", style="Heading 1")
        report.bullets(
            [
                "The downstream reference is the final expert-review workbook. "
                "Details of the review/consensus procedure are not fully encoded "
                "on every workbook row and should be described from the clinical "
                "study records in the manuscript.",
                "Four of 102 reviewed rows were excluded: one was not present in "
                "the annotation project and three were too difficult for a "
                "reliable clinical classification.",
                "The downstream analysis contains multiple FOVs from some source "
                "samples. Cluster-bootstrap intervals account for this grouping, "
                "but the number of represented samples remains 34.",
                "Confidence thresholds were calibrated on 640 × 640 object-level "
                "patches, whereas the downstream task uses overlapping SAHI "
                "slices extracted from full FOVs. The locked approach preserves "
                "test independence but does not remove this deployment-domain "
                "difference.",
                "The count-to-quality rule is deterministic and may be a larger "
                "source of downstream disagreement than residual detector error. "
                "This report does not tune that rule on the final test labels.",
                "Confidence intervals quantify resampling uncertainty in this "
                "dataset; they do not establish external generalizability.",
            ]
        )

        report.paragraph("9. Reproducibility and result artifacts", style="Heading 1")
        report.table(
            ["Artifact", "Location"],
            [
                ["Code commit", CODE_URL],
                ["Object-detection summary", str((object_output / "eval_summary.json").resolve())],
                ["Object predictions", str((object_output / "predictions_coco.json").resolve())],
                ["Calibration summary", str((calibration_output / "threshold_calibration_summary.json").resolve())],
                ["Calibration predictions", str((calibration_output / "calibration_predictions.json").resolve())],
                ["Downstream summary", str((downstream_output / "downstream_evaluation_summary.json").resolve())],
                ["Per-FOV downstream predictions", str((downstream_output / "downstream_predictions.csv").resolve())],
                ["Downstream exclusions", str((downstream_output / "downstream_exclusions.csv").resolve())],
                ["Downstream overlays", str((downstream_output / "overlays").resolve())],
            ],
            font_size=8,
        )
        report.paragraph(
            "The report was generated directly from the saved machine-readable "
            "artifacts above. All headline values can therefore be traced to a "
            "JSON or CSV result file."
        )

        report.page_break()
        report.paragraph("Appendix A. Downstream exclusions", style="Heading 1")
        exclusions = downstream["dataset"]["exclusions"]
        report.table(
            ["Workbook row", "Image", "Reason", "Comment"],
            [
                [
                    row["source_row"],
                    row["source_image_name"],
                    row["exclusion_reason"],
                    row["comment"] or "—",
                ]
                for row in exclusions
            ],
            font_size=8,
        )

        report.paragraph("Appendix B. Misclassified downstream FOVs", style="Heading 1")
        report.paragraph(
            "The table lists all downstream disagreements. Complete results for "
            "all 98 FOVs, including correct classifications and inference "
            "diagnostics, are retained in downstream_predictions.csv."
        )
        report.table(
            ["Image", "Clinical", "Predicted", "Leu", "Epi", "Rule score"],
            [
                [
                    row["image"],
                    row["clinical"],
                    row["predicted"],
                    row["leucocyte"],
                    row["epithelial"],
                    row["quality_score"],
                ]
                for row in descriptive["misclassified"]
            ],
            font_size=7.5,
        )

        statistics = report.save(report_path)
    except Exception:
        report.close_without_saving()
        raise

    return report_path, statistics


def validate_saved_report(path: Path, expected_statistics: dict[str, int]) -> None:
    if not path.is_file() or path.stat().st_size < 100_000:
        raise RuntimeError(f"Report was not created correctly: {path}")

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Open(str(path.resolve()), ReadOnly=True)
        document.Repaginate()
        pages = int(document.ComputeStatistics(WD_STATISTIC_PAGES))
        text = document.Content.Text
        required_phrases = [
            "Held-out object-detection evaluation",
            "Independent confidence-threshold calibration",
            "Final downstream quality evaluation",
            "0.5816",
            "0.5678",
        ]
        missing = [phrase for phrase in required_phrases if phrase not in text]
        if missing:
            raise RuntimeError(f"Report validation failed; missing text: {missing}")
        if pages < 8:
            raise RuntimeError(f"Report unexpectedly contains only {pages} pages.")
        if int(document.Tables.Count) != expected_statistics["tables"]:
            raise RuntimeError("Table count changed after reopening the report.")
        if int(document.InlineShapes.Count) != expected_statistics["figures"]:
            raise RuntimeError("Figure count changed after reopening the report.")
    finally:
        if document is not None:
            document.Close(SaveChanges=False)
        word.Quit()


def main() -> int:
    args = parse_args()
    report_path, statistics = build_report(args)
    validate_saved_report(report_path, statistics)
    print(
        json.dumps(
            {
                "report": str(report_path.resolve()),
                "size_bytes": report_path.stat().st_size,
                **statistics,
                "validation": "passed",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
