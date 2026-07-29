#!/usr/bin/env python
"""
calibrate_object_detection_thresholds_RFDETR.py

Calibrate class-specific RF-DETR object-detection score thresholds on an
independent COCO calibration set. This script is intentionally separate from
eval_object_detection_RFDETR.py: calibration chooses thresholds; evaluation
applies already-fixed thresholds to the final held-out test set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_path in (PROJECT_ROOT, SCRIPT_DIR):
    import_text = str(import_path)
    if import_text not in sys.path:
        sys.path.insert(0, import_text)

from eval_object_detection_RFDETR import (  # noqa: E402
    greedy_match,
    infer_model_class,
    iou_matrix,
    json_dump,
    load_model,
    predict_one_image,
    supported_rfdetr_model_names,
    timestamped_output_dir,
    write_csv,
)

try:
    import numpy as np
except Exception:  # pragma: no cover - dependency gate
    np = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover - dependency gate
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover - dependency gate
    Image = None  # type: ignore[assignment]


PYCHARM_USE_TOP_LEVEL_CONFIG = True

CALIBRATION_CLASS_NAMES = ["Leucocyte", "Squamous Epithelial Cell"]
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
PATCH_KEY_RE = re.compile(r"(BF\.\d+_\d+_patch_x\d+_y\d+)(?:\.[^.\\/]+)?$", re.IGNORECASE)

PYCHARM_CALIBRATE = {
    "calibration_root": r"E:\PHD\PhdData\CellScanData\Object Detection Calibration Dataset",
    "images_root": r"E:\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment",
    "run_dir": r"E:\PHD\Results\Quality Assessment\FINAL_B200\session_20260618_113853\TwoClass\HPO_Config_009",
    "checkpoint": r"E:\PHD\Results\Quality Assessment\FINAL_B200\session_20260618_113853\TwoClass\HPO_Config_009\checkpoint_best_ema.pth",
    "output_dir": r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\EvaluationOutput\RFDETR_ThresholdCalibration",
    "model_class": "auto",
    "score_floor": 0.001,
    "iou_threshold": 0.50,
    "threshold_min": 0.00,
    "threshold_max": 0.95,
    "threshold_step": 0.01,
    "selection_metric": "macro_f1",
    "max_images": None,
    "max_empty_images_per_task": 0,
    "seed": 42,
    "skip_missing_images": False,
    "metadata_only": False,
    "no_plots": False,
}


@dataclass
class CalibrationTask:
    name: str
    source: Path
    coco: Dict[str, Any]


@dataclass
class CalibrationConfig:
    calibration_root: Path
    images_root: Path
    run_dir: Path
    checkpoint: Path
    output_dir: Path
    model_class: str
    score_floor: float
    iou_threshold: float
    threshold_min: float
    threshold_max: float
    threshold_step: float
    selection_metric: str
    max_images: Optional[int]
    max_empty_images_per_task: Optional[int]
    seed: int
    skip_missing_images: bool
    metadata_only: bool
    no_plots: bool


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _append_optional_arg(argv: List[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        return
    text = str(value).strip()
    if text:
        argv.extend([flag, text])


def build_pycharm_argv() -> List[str]:
    cfg = PYCHARM_CALIBRATE
    argv: List[str] = []
    _append_optional_arg(argv, "--calibration-root", cfg.get("calibration_root"))
    _append_optional_arg(argv, "--images-root", cfg.get("images_root"))
    _append_optional_arg(argv, "--run-dir", cfg.get("run_dir"))
    _append_optional_arg(argv, "--checkpoint", cfg.get("checkpoint"))
    _append_optional_arg(argv, "--output-dir", cfg.get("output_dir"))
    _append_optional_arg(argv, "--model-class", cfg.get("model_class"))
    _append_optional_arg(argv, "--score-floor", cfg.get("score_floor"))
    _append_optional_arg(argv, "--iou-threshold", cfg.get("iou_threshold"))
    _append_optional_arg(argv, "--threshold-min", cfg.get("threshold_min"))
    _append_optional_arg(argv, "--threshold-max", cfg.get("threshold_max"))
    _append_optional_arg(argv, "--threshold-step", cfg.get("threshold_step"))
    _append_optional_arg(argv, "--selection-metric", cfg.get("selection_metric"))
    _append_optional_arg(argv, "--max-images", cfg.get("max_images"))
    _append_optional_arg(argv, "--max-empty-images-per-task", cfg.get("max_empty_images_per_task"))
    _append_optional_arg(argv, "--seed", cfg.get("seed"))
    _append_optional_arg(argv, "--skip-missing-images", cfg.get("skip_missing_images"))
    _append_optional_arg(argv, "--metadata-only", cfg.get("metadata_only"))
    _append_optional_arg(argv, "--no-plots", cfg.get("no_plots"))
    return argv


def ensure_dependencies(metadata_only: bool) -> None:
    missing: List[str] = []
    if np is None:
        missing.append("numpy")
    if Image is None:
        missing.append("Pillow")
    if not metadata_only and torch is None:
        missing.append("torch")
    if missing:
        raise ImportError("Missing dependencies: " + ", ".join(missing))


def canonical_class_index(name: str) -> Optional[int]:
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    if any(token in key for token in ("leucocyte", "leukocyte", "wbc")):
        return 0
    if any(token in key for token in ("epithelial", "squamous")):
        return 1
    return None


def normalize_prediction_class_ids(pred_labels: np.ndarray, n_classes: int) -> np.ndarray:
    if pred_labels.size == 0:
        return pred_labels.astype(np.int64)
    pred_labels = pred_labels.astype(np.int64)
    if np.all((pred_labels >= 0) & (pred_labels < n_classes)):
        return pred_labels
    if np.all((pred_labels >= 1) & (pred_labels <= n_classes)):
        return pred_labels - 1
    return np.clip(pred_labels, 0, max(0, n_classes - 1)).astype(np.int64)


def build_config(args: argparse.Namespace) -> CalibrationConfig:
    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    if args.model_class == "auto":
        model_class = infer_model_class(run_dir, checkpoint)
    else:
        model_class = str(args.model_class)

    output_dir = timestamped_output_dir(args.output_dir.resolve())
    if args.threshold_step <= 0:
        raise ValueError("--threshold-step must be > 0")
    if args.threshold_min < 0 or args.threshold_max > 1 or args.threshold_min >= args.threshold_max:
        raise ValueError("--threshold-min/--threshold-max must satisfy 0 <= min < max <= 1")
    return CalibrationConfig(
        calibration_root=args.calibration_root.resolve(),
        images_root=args.images_root.resolve(),
        run_dir=run_dir,
        checkpoint=checkpoint,
        output_dir=output_dir,
        model_class=model_class,
        score_floor=float(args.score_floor),
        iou_threshold=float(args.iou_threshold),
        threshold_min=float(args.threshold_min),
        threshold_max=float(args.threshold_max),
        threshold_step=float(args.threshold_step),
        selection_metric=str(args.selection_metric),
        max_images=int(args.max_images) if args.max_images is not None else None,
        max_empty_images_per_task=int(args.max_empty_images_per_task)
        if args.max_empty_images_per_task is not None
        else None,
        seed=int(args.seed),
        skip_missing_images=bool(args.skip_missing_images),
        metadata_only=bool(args.metadata_only),
        no_plots=bool(args.no_plots),
    )


def load_json_from_zip(path: Path) -> Dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        candidates = [name for name in archive.namelist() if name.lower().endswith(".json")]
        if not candidates:
            raise FileNotFoundError(f"No JSON annotation file found in {path}")
        preferred = [name for name in candidates if name.replace("\\", "/").endswith("annotations/instances_default.json")]
        json_name = preferred[0] if preferred else candidates[0]
        return json.loads(archive.read(json_name).decode("utf-8"))


def discover_tasks(calibration_root: Path) -> List[CalibrationTask]:
    tasks: List[CalibrationTask] = []
    for zip_path in sorted(calibration_root.glob("*.zip"), key=lambda p: p.name.lower()):
        tasks.append(CalibrationTask(name=zip_path.stem, source=zip_path, coco=load_json_from_zip(zip_path)))

    if tasks:
        return tasks

    for task_dir in sorted([p for p in calibration_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        json_candidates = sorted(task_dir.rglob("instances_default.json"))
        if not json_candidates:
            json_candidates = sorted(task_dir.rglob("*.json"))
        if not json_candidates:
            continue
        json_path = json_candidates[0]
        tasks.append(
            CalibrationTask(
                name=task_dir.name,
                source=json_path,
                coco=json.loads(json_path.read_text(encoding="utf-8")),
            )
        )
    return tasks


def index_task_images(images_root: Path, task_name: str) -> Dict[str, Path]:
    sample_dir = images_root / task_name
    preferred_patch_dir = sample_dir / f"patches for {task_name}"
    search_roots = [preferred_patch_dir] if preferred_patch_dir.exists() else []
    if sample_dir.exists() and sample_dir not in search_roots:
        search_roots.append(sample_dir)

    index: Dict[str, Path] = {}
    for root in search_roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.name, path)
                key = patch_key(path.name)
                if key:
                    index.setdefault(key, path)
                try:
                    rel = path.relative_to(sample_dir).as_posix()
                    index.setdefault(rel, path)
                except ValueError:
                    pass
    return index


def patch_key(file_name: str) -> Optional[str]:
    match = PATCH_KEY_RE.search(str(file_name).replace("\\", "/"))
    if not match:
        return None
    return match.group(1).lower()


def resolve_image_path(
    file_name: str,
    task_name: str,
    images_root: Path,
    task_index: Dict[str, Path],
) -> Path:
    raw = str(file_name).replace("\\", "/")
    basename = Path(raw).name
    candidates = [
        Path(raw),
        images_root / task_name / raw,
        images_root / task_name / f"patches for {task_name}" / raw,
        images_root / task_name / f"patches for {task_name}" / basename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if raw in task_index:
        return task_index[raw].resolve()
    if basename in task_index:
        return task_index[basename].resolve()
    key = patch_key(basename)
    if key and key in task_index:
        return task_index[key].resolve()
    raise FileNotFoundError(f"Could not resolve {file_name!r} for {task_name}")


def task_category_mapping(task: CalibrationTask) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for category in task.coco.get("categories", []):
        idx = canonical_class_index(str(category.get("name", "")))
        if idx is not None:
            mapping[int(category["id"])] = idx
    if not mapping:
        raise RuntimeError(f"No supported Leucocyte/Epithelial categories found in {task.source}")
    return mapping


def build_calibration_records(cfg: CalibrationConfig, tasks: Sequence[CalibrationTask]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    rng = random.Random(cfg.seed)

    for task in tasks:
        category_to_idx = task_category_mapping(task)
        image_by_id = {int(img["id"]): img for img in task.coco.get("images", [])}
        anns_by_image: Dict[int, List[Dict[str, Any]]] = {int(img_id): [] for img_id in image_by_id}
        for ann in task.coco.get("annotations", []):
            cat_id = int(ann.get("category_id", -1))
            if cat_id not in category_to_idx or int(ann.get("iscrowd", 0)) != 0:
                continue
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        image_ids_with_anns = [img_id for img_id, anns in anns_by_image.items() if anns]
        image_ids_without_anns = [img_id for img_id, anns in anns_by_image.items() if not anns]
        if cfg.max_empty_images_per_task is None or cfg.max_empty_images_per_task <= 0:
            image_ids_without_anns = []
        elif len(image_ids_without_anns) > cfg.max_empty_images_per_task:
            image_ids_without_anns = sorted(rng.sample(image_ids_without_anns, cfg.max_empty_images_per_task))

        task_image_ids = sorted(image_ids_with_anns) + sorted(image_ids_without_anns)
        task_index = index_task_images(cfg.images_root, task.name)

        for image_id in task_image_ids:
            info = image_by_id[image_id]
            file_name = str(info.get("file_name", ""))
            try:
                img_path = resolve_image_path(file_name, task.name, cfg.images_root, task_index)
            except FileNotFoundError as exc:
                payload = {"task": task.name, "image_id": image_id, "file_name": file_name, "error": str(exc)}
                if cfg.skip_missing_images:
                    missing.append(payload)
                    continue
                raise

            gt_boxes: List[List[float]] = []
            gt_cls: List[int] = []
            for ann in anns_by_image.get(image_id, []):
                x, y, w, h = ann["bbox"]
                gt_boxes.append([float(x), float(y), float(x + w), float(y + h)])
                gt_cls.append(category_to_idx[int(ann["category_id"])])

            records.append(
                {
                    "task": task.name,
                    "image_id": int(image_id),
                    "file_name": file_name,
                    "img_path": str(img_path),
                    "gt_boxes": np.asarray(gt_boxes, dtype=np.float32),
                    "gt_cls": np.asarray(gt_cls, dtype=np.int64),
                }
            )
            if cfg.max_images is not None and len(records) >= cfg.max_images:
                return records, missing

    return records, missing


def run_predictions(cfg: CalibrationConfig, records: Sequence[Dict[str, Any]]) -> None:
    model = load_model(cfg.model_class, cfg.checkpoint, cfg.run_dir)
    try:
        from tqdm import tqdm
        iterator = tqdm(records, desc="Calibrate infer", unit="img")
    except Exception:
        iterator = records

    inference_context = torch.inference_mode() if torch is not None else None
    context = inference_context if inference_context is not None else _NullContext()
    with context:
        for record in iterator:
            image = Image.open(record["img_path"]).convert("RGB")
            boxes, scores, pred_labels = predict_one_image(model, image, cfg.score_floor)
            record["pred_boxes"] = boxes.astype(np.float32)
            record["pred_scores"] = scores.astype(np.float32)
            record["pred_cls"] = normalize_prediction_class_ids(pred_labels, len(CALIBRATION_CLASS_NAMES))


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def threshold_grid(cfg: CalibrationConfig) -> np.ndarray:
    values = np.arange(
        cfg.threshold_min,
        cfg.threshold_max + cfg.threshold_step * 0.5,
        cfg.threshold_step,
        dtype=np.float32,
    )
    extras = np.asarray([0.30, 0.65], dtype=np.float32)
    values = np.unique(np.concatenate([values, extras]))
    values = values[(values >= cfg.threshold_min) & (values <= cfg.threshold_max)]
    return np.round(values, 6)


def count_for_class(
    records: Sequence[Dict[str, Any]],
    class_idx: int,
    threshold: float,
    iou_thr: float,
) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    for record in records:
        gt_mask = record["gt_cls"] == class_idx
        pred_mask = (record["pred_cls"] == class_idx) & (record["pred_scores"] >= threshold)
        gt_boxes = record["gt_boxes"][gt_mask]
        pred_boxes = record["pred_boxes"][pred_mask]

        matches = greedy_match(iou_matrix(gt_boxes, pred_boxes), iou_thr=iou_thr, valid_pairs=None)
        matched_gt = {gi for gi, _ in matches}
        matched_pr = {pj for _, pj in matches}
        tp += len(matches)
        fn += max(0, len(gt_boxes) - len(matched_gt))
        fp += max(0, len(pred_boxes) - len(matched_pr))
    return tp, fp, fn


def metrics_from_counts(tp: int, fp: int, fn: int, image_count: int) -> Dict[str, float]:
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
    jaccard = float(tp / (tp + fp + fn)) if (tp + fp + fn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "fp_per_image": float(fp / max(1, image_count)),
        "fn_per_image": float(fn / max(1, image_count)),
    }


def build_per_class_sweep(records: Sequence[Dict[str, Any]], thresholds: np.ndarray, iou_thr: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    image_count = len(records)
    for class_idx, class_name in enumerate(CALIBRATION_CLASS_NAMES):
        for threshold in thresholds.tolist():
            tp, fp, fn = count_for_class(records, class_idx, float(threshold), iou_thr)
            metric = metrics_from_counts(tp, fp, fn, image_count)
            rows.append(
                {
                    "class": class_name,
                    "threshold": float(threshold),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    **metric,
                }
            )
    return rows


def build_joint_sweep(per_class_rows: Sequence[Dict[str, Any]], selection_metric: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_class: Dict[str, List[Dict[str, Any]]] = {name: [] for name in CALIBRATION_CLASS_NAMES}
    for row in per_class_rows:
        by_class[str(row["class"])].append(row)

    joint_rows: List[Dict[str, Any]] = []
    for leu in by_class["Leucocyte"]:
        for epi in by_class["Squamous Epithelial Cell"]:
            tp = int(leu["tp"]) + int(epi["tp"])
            fp = int(leu["fp"]) + int(epi["fp"])
            fn = int(leu["fn"]) + int(epi["fn"])
            micro = metrics_from_counts(tp, fp, fn, image_count=1)
            row = {
                "leucocyte_threshold": float(leu["threshold"]),
                "epithelial_threshold": float(epi["threshold"]),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "micro_precision": micro["precision"],
                "micro_recall": micro["recall"],
                "micro_f1": micro["f1"],
                "micro_jaccard": micro["jaccard"],
                "macro_precision": float((float(leu["precision"]) + float(epi["precision"])) / 2.0),
                "macro_recall": float((float(leu["recall"]) + float(epi["recall"])) / 2.0),
                "macro_f1": float((float(leu["f1"]) + float(epi["f1"])) / 2.0),
                "macro_jaccard": float((float(leu["jaccard"]) + float(epi["jaccard"])) / 2.0),
                "leucocyte_precision": float(leu["precision"]),
                "leucocyte_recall": float(leu["recall"]),
                "leucocyte_f1": float(leu["f1"]),
                "leucocyte_jaccard": float(leu["jaccard"]),
                "leucocyte_fp_per_image": float(leu["fp_per_image"]),
                "epithelial_precision": float(epi["precision"]),
                "epithelial_recall": float(epi["recall"]),
                "epithelial_f1": float(epi["f1"]),
                "epithelial_jaccard": float(epi["jaccard"]),
                "epithelial_fp_per_image": float(epi["fp_per_image"]),
            }
            joint_rows.append(row)

    if not joint_rows:
        raise RuntimeError("No joint threshold rows were produced.")
    if selection_metric not in joint_rows[0]:
        raise ValueError(f"Unknown selection metric {selection_metric!r}")
    selected = max(joint_rows, key=lambda row: float(row[selection_metric]))
    return joint_rows, selected


def save_prediction_cache(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    payload: List[Dict[str, Any]] = []
    for record in records:
        payload.append(
            {
                "task": record["task"],
                "image_id": int(record["image_id"]),
                "file_name": record["file_name"],
                "img_path": record["img_path"],
                "gt_boxes": record["gt_boxes"].tolist(),
                "gt_cls": record["gt_cls"].tolist(),
                "pred_boxes": record["pred_boxes"].tolist(),
                "pred_scores": record["pred_scores"].tolist(),
                "pred_cls": record["pred_cls"].tolist(),
            }
        )
    json_dump(path, payload)


def maybe_save_plots(output_dir: Path, per_class_rows: Sequence[Dict[str, Any]], joint_rows: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    for class_name in CALIBRATION_CLASS_NAMES:
        rows = [row for row in per_class_rows if row["class"] == class_name]
        if not rows:
            continue
        rows.sort(key=lambda row: float(row["threshold"]))
        x = [float(row["threshold"]) for row in rows]
        plt.figure(figsize=(8, 5))
        for key in ("precision", "recall", "f1", "jaccard"):
            plt.plot(x, [float(row[key]) for row in rows], label=key)
        plt.xlabel("Score threshold")
        plt.ylabel("Metric")
        plt.title(f"{class_name} threshold calibration")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{class_name.lower().replace(' ', '_')}_threshold_sweep.png", dpi=180)
        plt.close()

    if joint_rows:
        thresholds_x = sorted({float(row["leucocyte_threshold"]) for row in joint_rows})
        thresholds_y = sorted({float(row["epithelial_threshold"]) for row in joint_rows})
        x_idx = {value: i for i, value in enumerate(thresholds_x)}
        y_idx = {value: i for i, value in enumerate(thresholds_y)}
        grid = np.full((len(thresholds_y), len(thresholds_x)), np.nan, dtype=np.float32)
        for row in joint_rows:
            grid[y_idx[float(row["epithelial_threshold"])]][x_idx[float(row["leucocyte_threshold"])]] = float(row["macro_f1"])
        plt.figure(figsize=(8, 6))
        plt.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[min(thresholds_x), max(thresholds_x), min(thresholds_y), max(thresholds_y)],
            vmin=0.0,
            vmax=1.0,
        )
        plt.colorbar(label="Macro F1")
        plt.xlabel("Leucocyte threshold")
        plt.ylabel("Epithelial threshold")
        plt.title("Joint threshold calibration")
        plt.tight_layout()
        plt.savefig(output_dir / "joint_macro_f1_heatmap.png", dpi=180)
        plt.close()


def run_calibration(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    ensure_dependencies(cfg.metadata_only)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    if not cfg.calibration_root.exists():
        raise FileNotFoundError(f"Calibration root not found: {cfg.calibration_root}")
    if not cfg.images_root.exists():
        raise FileNotFoundError(f"Images root not found: {cfg.images_root}")
    if not cfg.checkpoint.exists() and not cfg.metadata_only:
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(cfg.calibration_root)
    if not tasks:
        raise RuntimeError(f"No calibration task zips or JSON folders found under {cfg.calibration_root}")

    records, missing_images = build_calibration_records(cfg, tasks)
    annotation_counts = {
        class_name: int(sum(int(np.sum(record["gt_cls"] == idx)) for record in records))
        for idx, class_name in enumerate(CALIBRATION_CLASS_NAMES)
    }
    empty_images = int(sum(1 for record in records if len(record["gt_cls"]) == 0))

    print("[CALIBRATE] tasks       :", len(tasks))
    print("[CALIBRATE] images      :", len(records), f"(empty={empty_images})")
    print("[CALIBRATE] annotations :", annotation_counts)
    print("[CALIBRATE] output_dir  :", cfg.output_dir)

    metadata = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "calibration_root": str(cfg.calibration_root),
        "images_root": str(cfg.images_root),
        "run_dir": str(cfg.run_dir),
        "checkpoint": str(cfg.checkpoint),
        "model_class": cfg.model_class,
        "score_floor": cfg.score_floor,
        "iou_threshold": cfg.iou_threshold,
        "task_count": len(tasks),
        "image_count": len(records),
        "empty_image_count": empty_images,
        "annotation_counts": annotation_counts,
        "missing_images": missing_images,
    }
    json_dump(cfg.output_dir / "calibration_dataset_summary.json", metadata)

    if cfg.metadata_only:
        print("[CALIBRATE] Metadata-only mode; no model inference was run.")
        return

    run_predictions(cfg, records)
    save_prediction_cache(cfg.output_dir / "calibration_predictions.json", records)

    thresholds = threshold_grid(cfg)
    per_class_rows = build_per_class_sweep(records, thresholds, cfg.iou_threshold)
    joint_rows, selected = build_joint_sweep(per_class_rows, cfg.selection_metric)

    per_class_fieldnames = [
        "class",
        "threshold",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "jaccard",
        "fp_per_image",
        "fn_per_image",
    ]
    joint_fieldnames = [
        "leucocyte_threshold",
        "epithelial_threshold",
        "tp",
        "fp",
        "fn",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "micro_jaccard",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_jaccard",
        "leucocyte_precision",
        "leucocyte_recall",
        "leucocyte_f1",
        "leucocyte_jaccard",
        "leucocyte_fp_per_image",
        "epithelial_precision",
        "epithelial_recall",
        "epithelial_f1",
        "epithelial_jaccard",
        "epithelial_fp_per_image",
    ]
    write_csv(cfg.output_dir / "per_class_threshold_sweep.csv", per_class_fieldnames, per_class_rows)
    write_csv(cfg.output_dir / "joint_threshold_sweep.csv", joint_fieldnames, joint_rows)

    best_per_class = {}
    for class_name in CALIBRATION_CLASS_NAMES:
        rows = [row for row in per_class_rows if row["class"] == class_name]
        best_per_class[class_name] = max(rows, key=lambda row: float(row["f1"])) if rows else None

    selected_thresholds = {
        "Leucocyte": float(selected["leucocyte_threshold"]),
        "Squamous Epithelial Cell": float(selected["epithelial_threshold"]),
    }
    threshold_arg = ";".join(f"{name}={value:.4f}" for name, value in selected_thresholds.items())
    summary = {
        **metadata,
        "threshold_grid": {
            "min": cfg.threshold_min,
            "max": cfg.threshold_max,
            "step": cfg.threshold_step,
            "count": int(len(thresholds)),
        },
        "selection_metric": cfg.selection_metric,
        "selected_thresholds": selected_thresholds,
        "selected_threshold_arg": threshold_arg,
        "selected_joint_metrics": selected,
        "best_per_class_by_f1": best_per_class,
        "outputs": {
            "dataset_summary": str(cfg.output_dir / "calibration_dataset_summary.json"),
            "predictions": str(cfg.output_dir / "calibration_predictions.json"),
            "per_class_threshold_sweep": str(cfg.output_dir / "per_class_threshold_sweep.csv"),
            "joint_threshold_sweep": str(cfg.output_dir / "joint_threshold_sweep.csv"),
            "summary": str(cfg.output_dir / "threshold_calibration_summary.json"),
        },
    }
    json_dump(cfg.output_dir / "threshold_calibration_summary.json", summary)
    json_dump(cfg.output_dir / "selected_thresholds.json", selected_thresholds)

    if not cfg.no_plots:
        maybe_save_plots(cfg.output_dir, per_class_rows, joint_rows)

    print("[CALIBRATE] selected thresholds:", selected_thresholds)
    print("[CALIBRATE] evaluator argument :", f'--class-score-thresholds "{threshold_arg}"')
    print("[CALIBRATE] summary            :", (cfg.output_dir / "threshold_calibration_summary.json").resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate class-specific RF-DETR object detection thresholds.")
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-class",
        type=str,
        default=os.getenv("CALIB_MODEL_CLASS", "auto"),
        choices=supported_rfdetr_model_names(include_auto=True),
    )
    parser.add_argument("--score-floor", type=float, default=float(os.getenv("CALIB_SCORE_FLOOR", "0.001")))
    parser.add_argument("--iou-threshold", type=float, default=float(os.getenv("CALIB_IOU_THRESHOLD", "0.50")))
    parser.add_argument("--threshold-min", type=float, default=float(os.getenv("CALIB_THRESHOLD_MIN", "0.00")))
    parser.add_argument("--threshold-max", type=float, default=float(os.getenv("CALIB_THRESHOLD_MAX", "0.95")))
    parser.add_argument("--threshold-step", type=float, default=float(os.getenv("CALIB_THRESHOLD_STEP", "0.01")))
    parser.add_argument(
        "--selection-metric",
        type=str,
        default=os.getenv("CALIB_SELECTION_METRIC", "macro_f1"),
        choices=["macro_f1", "macro_jaccard", "micro_f1", "micro_jaccard"],
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--max-empty-images-per-task",
        type=int,
        default=0,
        help="Optional opt-in count of unannotated images per task. Default 0 excludes unannotated images.",
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("CALIB_SEED", "42")))
    parser.add_argument("--skip-missing-images", action="store_true", default=env_bool("CALIB_SKIP_MISSING_IMAGES", False))
    parser.add_argument("--metadata-only", action="store_true", default=env_bool("CALIB_METADATA_ONLY", False))
    parser.add_argument("--no-plots", action="store_true", default=env_bool("CALIB_NO_PLOTS", False))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    if argv is None and PYCHARM_USE_TOP_LEVEL_CONFIG and not sys.argv[1:]:
        argv = build_pycharm_argv()
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    run_calibration(args)


if __name__ == "__main__":
    main()
