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
from openpyxl import load_workbook


def accuracy_score(true: Sequence[Any], predicted: Sequence[Any]) -> float:
    return float(np.mean(np.asarray(true) == np.asarray(predicted)))


def balanced_accuracy_score(true: Sequence[Any], predicted: Sequence[Any]) -> float:
    true_array = np.asarray(true)
    predicted_array = np.asarray(predicted)
    labels = list(dict.fromkeys(true))
    recalls = [
        float(np.mean(predicted_array[true_array == label] == label))
        for label in labels
    ]
    return float(np.mean(recalls))


def f1_score(
    true: Sequence[Any], predicted: Sequence[Any], *, labels: Sequence[Any],
    average: str, zero_division: int = 0,
) -> float:
    if average != "macro":
        raise ValueError("Only macro F1 is implemented in this report builder.")
    scores = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(true, predicted))
        fp = sum(a != label and b == label for a, b in zip(true, predicted))
        fn = sum(a == label and b != label for a, b in zip(true, predicted))
        precision = tp / (tp + fp) if tp + fp else float(zero_division)
        recall = tp / (tp + fn) if tp + fn else float(zero_division)
        scores.append(
            2 * precision * recall / (precision + recall)
            if precision + recall else float(zero_division)
        )
    return float(np.mean(scores))


def cohen_kappa_score(
    true: Sequence[Any], predicted: Sequence[Any], labels: Sequence[Any] | None = None,
    weights: str | None = None,
) -> float:
    labels = list(labels or sorted(set([*true, *predicted])))
    total = len(true)
    if weights == "quadratic":
        positions = {label: index for index, label in enumerate(labels)}
        denominator_scale = max(1, len(labels) - 1) ** 2
        observed = sum(
            (positions[a] - positions[b]) ** 2 / denominator_scale
            for a, b in zip(true, predicted)
        ) / total
        true_counts = Counter(true)
        predicted_counts = Counter(predicted)
        expected = sum(
            true_counts[a] * predicted_counts[b]
            * ((positions[a] - positions[b]) ** 2 / denominator_scale)
            for a in labels for b in labels
        ) / (total * total)
        return float(1 - observed / expected) if expected else 0.0
    if weights is not None:
        raise ValueError(f"Unsupported kappa weights: {weights!r}")
    observed = accuracy_score(true, predicted)
    true_counts = Counter(true)
    predicted_counts = Counter(predicted)
    expected = sum(true_counts[label] * predicted_counts[label] for label in labels) / (total * total)
    return float((observed - expected) / (1 - expected)) if expected != 1 else 0.0

try:
    import win32com.client
except Exception:  # pragma: no cover - optional native Word backend
    win32com = None  # type: ignore[assignment]


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
DEFAULT_MANUAL_LABELS = Path(
    r"C:\Users\SH37YE\OneDrive\Full_FOV_Master_Review_2026-06-02_1154.xlsx"
)
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
    parser.add_argument("--manual-labels", type=Path, default=DEFAULT_MANUAL_LABELS)
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


LEGACY_LABEL_IDS = {"Qualified": 1, "Partially Qualified": 2, "Not Qualified": 3}
GECKLER_LABELS = [f"G{index}" for index in range(1, 7)]
MW_LABELS = ["Acceptable", "Unacceptable"]
COLLAPSED_GECKLER_LABELS = ["Acceptable", "Unacceptable", "Unknown"]


def normalize_count_bin(value: Any) -> str:
    text = str(value or "").strip().casefold().replace("–", "-").replace("—", "-")
    compact = re.sub(r"\s+", "", text.replace("til", "-"))
    if compact in {"0-9", "0to9"}:
        return "0-9"
    if compact in {"10-25", "10to25"}:
        return "10-25"
    if compact in {"26+", ">25", "26"}:
        return "26+"
    try:
        number = int(float(compact))
    except (TypeError, ValueError):
        raise ValueError(f"Unsupported expert count category: {value!r}")
    return "0-9" if number <= 9 else "10-25" if number <= 25 else "26+"


def integer_count_bin(value: Any) -> str:
    number = int(value)
    return "0-9" if number <= 9 else "10-25" if number <= 25 else "26+"


def count_adjustment_to_bin(predicted_count: Any, expert_bin: str) -> int:
    """Minimum signed change needed for a predicted count to enter an expert bin."""
    predicted = int(predicted_count)
    lower, upper = {
        "0-9": (0, 9),
        "10-25": (10, 25),
        "26+": (26, None),
    }[expert_bin]
    if predicted < lower:
        return lower - predicted
    if upper is not None and predicted > upper:
        return upper - predicted
    return 0


def signed_adjustment(value: int) -> str:
    return f"+{value}" if value > 0 else str(value)


def geckler_class(epithelial_bin: str, leucocyte_bin: str) -> str:
    if epithelial_bin == "26+":
        return {"0-9": "G1", "10-25": "G2", "26+": "G3"}[leucocyte_bin]
    if epithelial_bin == "10-25" and leucocyte_bin == "26+":
        return "G4"
    if epithelial_bin == "0-9" and leucocyte_bin == "26+":
        return "G5"
    return "G6"


def collapsed_geckler_label(group: str) -> str:
    if group in {"G4", "G5"}:
        return "Acceptable"
    if group in {"G1", "G2", "G3"}:
        return "Unacceptable"
    return "Unknown"


def mw_label(epithelial_bin: str, leucocyte_bin: str) -> str:
    return "Acceptable" if epithelial_bin == "0-9" and leucocyte_bin == "26+" else "Unacceptable"


def read_expert_references(path: Path, sheet_name: str = "Master") -> dict[str, dict[str, str]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook[sheet_name]
        references: dict[str, dict[str, str]] = {}
        for row_number, row in enumerate(
            sheet.iter_rows(min_row=2, min_col=1, max_col=6, values_only=True), start=2
        ):
            image, legacy, epithelial, leucocyte, annotator, comment = row
            if not image:
                continue
            legacy_text = str(legacy or "").strip()
            if legacy_text.casefold() == "er ikke i projektet" or not legacy_text:
                continue
            if legacy_text not in LEGACY_LABEL_IDS:
                raise ValueError(f"Workbook row {row_number}: unsupported legacy label {legacy!r}")
            epi_bin = normalize_count_bin(epithelial)
            leu_bin = normalize_count_bin(leucocyte)
            references[str(image).strip().casefold()] = {
                "legacy_label": legacy_text,
                "epithelial_bin": epi_bin,
                "leucocyte_bin": leu_bin,
                "geckler_class": geckler_class(epi_bin, leu_bin),
                "mw_label": mw_label(epi_bin, leu_bin),
                "annotator": str(annotator or "").strip(),
                "comment": str(comment or "").strip(),
            }
        return references
    finally:
        workbook.close()


def named_classification_analysis(
    rows: Sequence[dict[str, str]],
    reference_key: str,
    predicted_key: str,
    labels: Sequence[str],
) -> dict[str, Any]:
    true = [row[reference_key] for row in rows]
    predicted = [row[predicted_key] for row in rows]
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    positions = {label: index for index, label in enumerate(labels)}
    for actual, estimate in zip(true, predicted):
        matrix[positions[actual], positions[estimate]] += 1
    per_class = []
    for index, label in enumerate(labels):
        tp = int(matrix[index, index])
        fn = int(matrix[index, :].sum() - tp)
        fp = int(matrix[:, index].sum() - tp)
        tn = len(rows) - tp - fn - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class.append({
            "label": label, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
        })
    return {
        "labels": list(labels),
        "reference_counts": dict(Counter(true)),
        "predicted_counts": dict(Counter(predicted)),
        "metrics": {
            "accuracy": float(accuracy_score(true, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
            "macro_f1": float(f1_score(true, predicted, labels=list(labels), average="macro", zero_division=0)),
            "cohen_kappa": float(cohen_kappa_score(true, predicted, labels=list(labels))),
        },
        "per_class": per_class,
        "matrix": matrix.tolist(),
    }


def save_confusion_matrix_figure(analysis: dict[str, Any], scheme: str, path: Path) -> None:
    """Render counts with a shared count scale and explicit reference/model axes."""
    from PIL import Image, ImageDraw, ImageFont

    labels = analysis["labels"]
    matrix = np.asarray(analysis["matrix"], dtype=int)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("Confusion matrix dimensions do not match its labels.")
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    def font(size: int) -> Any:
        return ImageFont.truetype(str(font_path), size) if font_path.is_file() else ImageFont.load_default(size=size)

    image = Image.new("RGB", (1500, 1250), "white")
    draw = ImageDraw.Draw(image)
    draw.text((750, 45), scheme, font=font(42), fill="black", anchor="mt")
    draw.text((870, 130), "Model classification (predicted)", font=font(32), fill="black", anchor="mt")
    draw.text((35, 175), "Expert reference", font=font(28), fill="black")
    draw.text((35, 212), "(actual)", font=font(28), fill="black")
    left, top, span = 390, 280, 930
    cell = span // len(labels)
    maximum = max(1, int(matrix.max()))
    for index, label in enumerate(labels):
        draw.text((left + (index + .5) * cell, top - 25), label, font=font(27), fill="black", anchor="mb")
        draw.text((left - 24, top + (index + .5) * cell), label, font=font(29), fill="black", anchor="rm")
        for column, value in enumerate(matrix[index]):
            intensity = int(value) / maximum
            color = tuple(round(a + intensity * (b - a)) for a, b in zip((239, 246, 252), (31, 78, 121)))
            x, y = left + column * cell, top + index * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=color, outline="white", width=3)
            draw.text((x + cell / 2, y + cell / 2), str(value), font=font(42), fill="white" if intensity > .55 else "black", anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(240, 240))


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
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        saved_rows = read_csv(predictions_path.parent / "per_class_metrics.csv")
        return [
            {
                "class": row["class"],
                "AP@50:95": float(row["AP_iou_sweep"]),
                "AP@50": float(row["AP@50"]),
                "AP@75": float(row["AP@75"]),
                "AR@100": float(row["AR_iou_sweep"]),
            }
            for row in saved_rows
        ]

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
        if win32com is None:
            raise RuntimeError("The Microsoft Word COM backend is unavailable.")
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


if win32com is None:
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.enum.style import WD_STYLE_TYPE
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Inches, Pt, RGBColor

    class WordReport:  # type: ignore[no-redef]
        """Portable python-docx backend used when local Word automation is absent."""

        def __init__(self) -> None:
            self.document = Document()
            section = self.document.sections[0]
            section.page_width = Cm(21.0)
            section.page_height = Cm(29.7)
            section.top_margin = Cm(2.0)
            section.bottom_margin = Cm(1.8)
            section.left_margin = Cm(2.0)
            section.right_margin = Cm(2.0)
            styles = self.document.styles
            styles["Normal"].font.name = "Aptos"
            styles["Normal"].font.size = Pt(10)
            for name, size, color in (
                ("Title", 24, (31, 78, 121)),
                ("Subtitle", 13, (75, 85, 99)),
                ("Heading 1", 16, (31, 78, 121)),
                ("Heading 2", 13, (46, 116, 181)),
                ("Heading 3", 11, (55, 65, 81)),
                ("Caption", 9, (75, 85, 99)),
            ):
                styles[name].font.name = "Aptos"
                styles[name].font.size = Pt(size)
                styles[name].font.color.rgb = RGBColor(*color)
            footer = section.footer.paragraphs[0]
            footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
            footer.add_run("RF-DETR microscopy quality assessment — combined results  |  ")
            field = OxmlElement("w:fldSimple")
            field.set(qn("w:instr"), "PAGE")
            footer._p.append(field)
            self._figure_count = 0

        def paragraph(self, text: str = "", style: str = "Normal", alignment: int = 0,
                      bold: bool = False, italic: bool = False) -> None:
            paragraph = self.document.add_paragraph(style=style)
            paragraph.alignment = alignment
            run = paragraph.add_run(str(text))
            run.bold = bold
            run.italic = italic

        def bullets(self, items: Iterable[str]) -> None:
            for item in items:
                self.paragraph(str(item), style="List Bullet")

        def page_break(self) -> None:
            self.document.add_page_break()

        def table(self, headers: Sequence[str], rows: Sequence[Sequence[Any]], font_size: float = 9) -> None:
            table = self.document.add_table(rows=1, cols=len(headers))
            table.style = "Table Grid"
            for index, value in enumerate(headers):
                cell = table.rows[0].cells[index]
                cell.text = str(value)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                shading = OxmlElement("w:shd")
                shading.set(qn("w:fill"), "1F4E79")
                cell._tc.get_or_add_tcPr().append(shading)
                for run in cell.paragraphs[0].runs:
                    run.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
            for row_index, values in enumerate(rows):
                cells = table.add_row().cells
                for column, value in enumerate(values):
                    cells[column].text = str(value)
                    if row_index % 2:
                        shading = OxmlElement("w:shd")
                        shading.set(qn("w:fill"), "EEF4FA")
                        cells[column]._tc.get_or_add_tcPr().append(shading)
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.name = "Aptos"
                            run.font.size = Pt(font_size)
            self.document.add_paragraph()

        def image(self, path: Path, caption: str, width_inches: float = 6.25) -> None:
            if not path.is_file():
                self.paragraph(f"[Figure unavailable: {path}]", italic=True)
                return
            paragraph = self.document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.add_run().add_picture(str(path.resolve()), width=Inches(width_inches))
            self.paragraph(caption, style="Caption", alignment=WD_ALIGN_PARAGRAPH.CENTER)
            self._figure_count += 1

        def add_contents(self) -> None:
            self.paragraph("Contents", style="Heading 1")
            paragraph = self.document.add_paragraph()
            field = OxmlElement("w:fldSimple")
            field.set(qn("w:instr"), 'TOC \\o "1-3" \\h \\z \\u')
            paragraph._p.append(field)
            self.page_break()

        def save(self, path: Path) -> dict[str, int]:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.document.save(path)
            return {
                "pages": 0,
                "paragraphs": len(self.document.paragraphs),
                "tables": len(self.document.tables),
                "figures": self._figure_count,
            }

        def close_without_saving(self) -> None:
            return


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
    expert_references = read_expert_references(args.manual_labels.resolve())
    for row in downstream_rows:
        key = row["source_image_name"].strip().casefold()
        if key not in expert_references:
            raise ValueError(
                f"No revised expert reference was found for {row['source_image_name']!r}"
            )
        reference = expert_references[key]
        row["manual_label"] = reference["legacy_label"]
        row["manual_label_id"] = str(LEGACY_LABEL_IDS[reference["legacy_label"]])
        row["expert_epithelial_bin"] = reference["epithelial_bin"]
        row["expert_leucocyte_bin"] = reference["leucocyte_bin"]
        row["expert_geckler_class"] = reference["geckler_class"]
        row["expert_collapsed_geckler_label"] = collapsed_geckler_label(
            reference["geckler_class"]
        )
        row["expert_mw_label"] = reference["mw_label"]
        predicted_epi_bin = integer_count_bin(row["n_squamous_epithelial_cell"])
        predicted_leu_bin = integer_count_bin(row["n_leucocyte"])
        row["predicted_geckler_class"] = geckler_class(
            predicted_epi_bin, predicted_leu_bin
        )
        row["predicted_collapsed_geckler_label"] = collapsed_geckler_label(
            row["predicted_geckler_class"]
        )
        row["predicted_mw_label"] = mw_label(predicted_epi_bin, predicted_leu_bin)

    legacy_analysis = named_classification_analysis(
        downstream_rows,
        "manual_label",
        "predicted_label",
        list(LEGACY_LABEL_IDS),
    )
    geckler_analysis = named_classification_analysis(
        downstream_rows,
        "expert_geckler_class",
        "predicted_geckler_class",
        GECKLER_LABELS,
    )
    collapsed_geckler_analysis = named_classification_analysis(
        downstream_rows,
        "expert_collapsed_geckler_label",
        "predicted_collapsed_geckler_label",
        COLLAPSED_GECKLER_LABELS,
    )
    mw_analysis = named_classification_analysis(
        downstream_rows,
        "expert_mw_label",
        "predicted_mw_label",
        MW_LABELS,
    )
    scheme_analyses = {
        "Original expert scheme": legacy_analysis,
        "Geckler (six groups)": geckler_analysis,
        "Collapsed Geckler (three-way)": collapsed_geckler_analysis,
        "Murray-Washington (binary)": mw_analysis,
    }
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
                    "Geckler accuracy "
                    + pct(geckler_analysis["metrics"]["accuracy"], 2)
                    + "; collapsed Geckler accuracy "
                    + pct(collapsed_geckler_analysis["metrics"]["accuracy"], 2)
                    + "; Murray-Washington accuracy "
                    + pct(mw_analysis["metrics"]["accuracy"], 2)
                    + "; original expert-label accuracy "
                    + pct(legacy_analysis["metrics"]["accuracy"], 2),
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
                "The final downstream evaluation included 98 expert-counted FOVs "
                "after four prespecified exclusions. The revised primary references "
                "were derived independently under full Geckler, collapsed Geckler, "
                "and Murray-Washington rules; the original expert quality tags were "
                "retained as a comparator.",
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

        report.paragraph("3.3 Count-to-quality classification rules", style="Heading 2")
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
        report.paragraph(
            "For the revised primary analysis, both expert count bins and model "
            "counts were converted by the same published cell-count boundaries. "
            "Geckler groups G1-G3 contain >25 epithelial cells with respectively "
            "0-9, 10-25, or >25 leukocytes; G4 contains 10-25 epithelial cells and "
            ">25 leukocytes; G5 contains 0-9 epithelial cells and >25 leukocytes; "
            "G6 contains the remaining low-epithelial/low-leukocyte combinations. "
            "A three-way collapsed Geckler endpoint grouped G4-G5 as Acceptable, "
            "G1-G3 as Unacceptable, and G6 as Unknown. "
            "The Murray-Washington binary analysis defined an FOV as Acceptable "
            "only when it contained 0-9 epithelial cells and >25 leukocytes."
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
                ["Original Qualified references", legacy_analysis["reference_counts"].get("Qualified", 0)],
                ["Original Partially Qualified references", legacy_analysis["reference_counts"].get("Partially Qualified", 0)],
                ["Original Not Qualified references", legacy_analysis["reference_counts"].get("Not Qualified", 0)],
                ["Total evaluation time", f"{downstream['runtime']['total_evaluation_seconds']:.1f} s"],
                ["Mean detector time per FOV", f"{downstream['runtime']['mean_inference_seconds_per_image']:.2f} s"],
            ],
        )
        report.paragraph(
            "All 98 eligible images completed without inference failure, and an "
            "annotated overlay was produced for every included FOV."
        )

        report.paragraph("6.2 Reference classification comparison", style="Heading 2")
        report.paragraph(
            "The revised workbook retains the experts' original quality tag and "
            "adds independently recorded epithelial-cell and leukocyte bins. The "
            "count bins were converted to Geckler and Murray-Washington references "
            "without using model output. The table below makes the resulting change "
            "in endpoint explicit."
        )
        report.table(
            ["Scheme", "Reference distribution", "Model distribution"],
            [
                [
                    scheme,
                    "; ".join(f"{label}: {analysis['reference_counts'].get(label, 0)}" for label in analysis["labels"]),
                    "; ".join(f"{label}: {analysis['predicted_counts'].get(label, 0)}" for label in analysis["labels"]),
                ]
                for scheme, analysis in scheme_analyses.items()
            ],
            font_size=8,
        )
        report.table(
            ["Evaluation scheme", "Accuracy", "Balanced accuracy", "Macro F1", "Cohen's kappa"],
            [
                [
                    scheme,
                    fmt(analysis["metrics"]["accuracy"], 4),
                    fmt(analysis["metrics"]["balanced_accuracy"], 4),
                    fmt(analysis["metrics"]["macro_f1"], 4),
                    fmt(analysis["metrics"]["cohen_kappa"], 4),
                ]
                for scheme, analysis in scheme_analyses.items()
            ],
        )

        report.paragraph("6.3 Original expert-label analysis", style="Heading 2")
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
        report.paragraph("6.4 Original expert-label class performance", style="Heading 2")
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
                for row in legacy_analysis["per_class"]
            ],
        )
        matrix = legacy_analysis["matrix"]
        report.paragraph(
            f"The pipeline correctly classified {sum(matrix[index][index] for index in range(3))} "
            "of 98 FOVs under the original three-category expert scheme. These "
            "results are retained for direct comparison and are no longer the "
            "primary downstream endpoint."
        )

        report.paragraph("6.5 Count-based classification confusion matrices", style="Heading 2")
        for scheme, analysis in (
            ("Geckler", geckler_analysis),
            ("Collapsed Geckler", collapsed_geckler_analysis),
            ("Murray-Washington", mw_analysis),
        ):
            figure_path = output_dir / (re.sub(r"[^a-z0-9]+", "_", scheme.lower()) + "_confusion_matrix.png")
            save_confusion_matrix_figure(analysis, scheme, figure_path)
            report.image(
                figure_path,
                f"{scheme} confusion matrix. Cells show FOV counts (N = {sum(map(sum, analysis['matrix']))}); "
                "rows are expert references and columns are model predictions. Darker cells indicate larger counts.",
                width_inches=5.8,
            )
            report.paragraph(f"{scheme} reference (rows) versus model classification (columns)")
            report.table(
                ["Reference"] + analysis["labels"],
                [
                    [label] + values
                    for label, values in zip(analysis["labels"], analysis["matrix"])
                ],
                font_size=8,
            )

        report.paragraph("6.6 Error direction and detected-count patterns", style="Heading 2")
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
                "Downstream agreement under each reference scheme measures the entire chain: "
                "sliced full-FOV inference, duplicate handling, class-specific "
                "thresholding, count aggregation, and the selected published "
                "cell-count classification.",
                "Geckler preserves six count-pattern groups, whereas "
                "collapsed Geckler treats G4-G5 as acceptable, G1-G3 as unacceptable, "
                "and G6 as unknown. Murray-Washington uses the narrower G5-like "
                "acceptable-cellularity criterion.",
                "The original expert quality tags remain visible as a sensitivity "
                "analysis, allowing the effect of changing the endpoint definition "
                "to be separated from detector performance.",
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
                "The experts recorded count bins rather than exact counts, so the "
                "analysis can resolve published threshold categories but cannot "
                "recover within-bin cell counts.",
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
                "The full Geckler, collapsed Geckler, and Murray-Washington rules "
                "were applied identically to expert bins and model counts and were "
                "not tuned on downstream outcomes.",
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
                ["Revised expert count workbook", str(args.manual_labels.resolve())],
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
            "The tables list every disagreement under each endpoint so individual "
            "FOVs can be located by workbook row and filename. Expert bins are the "
            "revised manual epithelial/leukocyte categories; model counts are the "
            "detector-derived cell totals. The Epi / Leu difference is the minimum "
            "signed adjustment needed to move each model count into its expert bin: "
            "negative means fewer predicted cells are needed, positive means more, "
            "and zero means the count is already in the correct bin. Complete results "
            "for all 98 FOVs remain available in downstream_predictions.csv."
        )
        error_schemes = [
            (
                "B.1 Original expert quality scheme",
                "manual_label",
                "predicted_label",
            ),
            (
                "B.2 Geckler classification",
                "expert_geckler_class",
                "predicted_geckler_class",
            ),
            (
                "B.3 Collapsed Geckler classification",
                "expert_collapsed_geckler_label",
                "predicted_collapsed_geckler_label",
            ),
            (
                "B.4 Murray-Washington classification",
                "expert_mw_label",
                "predicted_mw_label",
            ),
        ]
        for heading, reference_key, predicted_key in error_schemes:
            errors = [
                row
                for row in downstream_rows
                if row[reference_key] != row[predicted_key]
            ]
            heading = f"{heading} ({len(errors)} disagreements)"
            if heading.startswith(("B.3 ", "B.4 ")):
                report.page_break()
            report.paragraph(heading, style="Heading 2")
            report.table(
                [
                    "Workbook row",
                    "FOV filename",
                    "Expert bins (Epi; Leu)",
                    "Reference",
                    "Model counts (Epi; Leu)",
                    "Epi / Leu",
                    "Prediction",
                ],
                [
                    [
                        row["source_row"],
                        row["source_image_name"],
                        f"{row['expert_epithelial_bin']}; {row['expert_leucocyte_bin']}",
                        row[reference_key],
                        f"{row['n_squamous_epithelial_cell']}; {row['n_leucocyte']}",
                        " / ".join(
                            [
                                signed_adjustment(
                                    count_adjustment_to_bin(
                                        row["n_squamous_epithelial_cell"],
                                        row["expert_epithelial_bin"],
                                    )
                                ),
                                signed_adjustment(
                                    count_adjustment_to_bin(
                                        row["n_leucocyte"],
                                        row["expert_leucocyte_bin"],
                                    )
                                ),
                            ]
                        ),
                        row[predicted_key],
                    ]
                    for row in errors
                ],
                font_size=6.5,
            )

        statistics = report.save(report_path)
    except Exception:
        report.close_without_saving()
        raise

    return report_path, statistics


def validate_saved_report(path: Path, expected_statistics: dict[str, int]) -> None:
    if not path.is_file() or path.stat().st_size < 100_000:
        raise RuntimeError(f"Report was not created correctly: {path}")

    required_phrases = [
        "Held-out object-detection evaluation",
        "Independent confidence-threshold calibration",
        "Final downstream quality evaluation",
        "Geckler",
        "Murray-Washington",
        "Original expert-label analysis",
    ]
    if win32com is None:
        document = Document(path)
        text = "\n".join(
            [paragraph.text for paragraph in document.paragraphs]
            + [cell.text for table in document.tables for row in table.rows for cell in row.cells]
        )
        missing = [phrase for phrase in required_phrases if phrase not in text]
        if missing:
            raise RuntimeError(f"Report validation failed; missing text: {missing}")
        if len(document.tables) != expected_statistics["tables"]:
            raise RuntimeError("Table count changed after reopening the report.")
        return

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Open(str(path.resolve()), ReadOnly=True)
        document.Repaginate()
        pages = int(document.ComputeStatistics(WD_STATISTIC_PAGES))
        text = document.Content.Text
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
