#!/usr/bin/env python
"""
Local SAHI-based full-FOV RF-DETR evaluation script.

Purpose:
- run sliced inference on large full-FOV images using SAHI
- aggregate prediction counts per image and per sample
- optionally evaluate against COCO GT once full-FOV labels are ready

Notes:
- this script is intentionally separate from EvalRFDETR_SOLO_LocalOnly.py
- it reuses the local RF-DETR loading/prediction helpers from that script
- it currently targets the "Sample 71 and forward" full-FOV workflow
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:
    np = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import EvalRFDETR_SOLO_LocalOnly as local_eval


ACTIVE_PRESET = "two_class"

DEFAULT_IMAGES_ROOT = r"D:\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment_Cleaned"
DEFAULT_OUTPUT_DIR = r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\FOV_SAHI_Output"
DEFAULT_SAMPLE_MIN = 71
SAMPLE_SELECTION = "73" # "All" = all, "71-80" inclusive range, "75+" from 75 and up, "88" = only 88
OVERLAY_SELECTION = "25"
CLASS_SCORE_THRESHOLDS: Dict[str, float] = {
    "Leucocyte": 0.30,
    "Squamous Epithelial Cell": 0.70,
}

DEFAULT_CLASS_NAMES = [
    "Leucocyte",
    "Squamous Epithelial Cell",
]

FOV_PRESETS = {
    "two_class": {
        "target_name": "TwoClass",
        "checkpoint": (
            r"D:\PHD\Results\Quality Assessment\Epi+Leu for ESCMID Conference\second full two class - new data"
            r"\session_20260327_000629\TwoClass\HPO_Config_001\checkpoint_best_total.pth"
        ),
        "images_root": DEFAULT_IMAGES_ROOT,
        "model_class": "auto",
        "class_names": DEFAULT_CLASS_NAMES,
    },
}


@dataclass
class FOVSahiConfig:
    preset_name: str
    checkpoint: Path
    images_root: Path
    output_dir: Path
    output_root_dir: Path
    run_timestamp: str
    target_name: str
    class_names: List[str]
    gt_json: Optional[Path]
    model_class: str
    model_resolution: Optional[int]
    sample_min: int
    sample_max: Optional[int]
    sample_selection: str
    score_floor: float
    score_threshold: float
    class_score_thresholds: Dict[str, float]
    confmat_iou: float
    seed: int
    progress_every: int
    num_overlays: int
    no_plots: bool
    slice_height: Optional[int]
    slice_width: Optional[int]
    overlap_height_ratio: float
    overlap_width_ratio: float
    perform_standard_pred: bool
    postprocess_type: str
    postprocess_match_metric: str
    postprocess_match_threshold: float
    postprocess_class_agnostic: bool


def _optional_path(raw: str) -> Optional[Path]:
    raw = (raw or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _resolve_preset(name: str) -> Dict[str, Any]:
    key = (name or "").strip().lower()
    if key not in FOV_PRESETS:
        raise ValueError(
            f"Unknown preset {name!r}. Available presets: {', '.join(sorted(FOV_PRESETS))}"
        )
    return FOV_PRESETS[key]


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_output_dir(base_output_dir: Path, target_name: str) -> Tuple[Path, Path, str]:
    suffix = re.sub(r"[^A-Za-z0-9]+", "", target_name) or "FOV"
    root_dir = base_output_dir.parent / f"{base_output_dir.name}_{suffix}"
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = root_dir / run_stamp
    counter = 2
    while out_dir.exists():
        out_dir = root_dir / f"{run_stamp}_{counter:02d}"
        counter += 1
    return out_dir, root_dir, out_dir.name


def infer_checkpoint_runtime_metadata(checkpoint: Path) -> Tuple[Optional[int], Optional[List[str]]]:
    if local_eval.torch is None:
        return None, None
    try:
        ckpt = local_eval.torch.load(  # type: ignore[union-attr]
            str(checkpoint),
            map_location="cpu",
            weights_only=False,
        )
    except Exception:
        return None, None

    args = ckpt.get("args")
    if args is None:
        return None, None

    num_classes = None
    class_names = None
    try:
        num_classes = int(getattr(args, "num_classes", None))
    except Exception:
        num_classes = None
    try:
        raw_names = getattr(args, "class_names", None)
        if raw_names:
            class_names = [str(name).strip() for name in raw_names if str(name).strip()]
    except Exception:
        class_names = None
    return num_classes, class_names


def load_model_for_fov(
    model_class: str,
    checkpoint: Path,
    resolution: Optional[int],
    num_classes: Optional[int],
    class_names: Optional[Sequence[str]],
) -> Any:
    local_eval.ensure_rfdetr_import()
    from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge

    name_to_cls = {
        "RFDETRSmall": RFDETRSmall,
        "RFDETRMedium": RFDETRMedium,
        "RFDETRLarge": RFDETRLarge,
    }
    if model_class not in name_to_cls:
        raise ValueError(f"Unsupported model_class={model_class}")

    kwargs: Dict[str, Any] = {"pretrain_weights": str(checkpoint)}
    if resolution is not None:
        kwargs["resolution"] = int(resolution)
    if num_classes is not None:
        kwargs["num_classes"] = int(num_classes)

    model = name_to_cls[model_class](**kwargs)
    if class_names:
        try:
            model.model.class_names = list(class_names)
        except Exception:
            pass
    if hasattr(model, "optimize_for_inference"):
        try:
            model.optimize_for_inference()
        except Exception:
            pass
    return model


def parse_overlay_selection(raw: Any) -> int:
    txt = str(raw).strip().lower()
    if txt in {"all", "*"}:
        return -1
    if txt in {"none", "off", "0"}:
        return 0
    value = int(txt)
    if value < 0:
        raise ValueError("Overlay selection must be 'all', '0', or a positive integer.")
    return value


def ensure_deps() -> None:
    local_eval.ensure_deps()
    missing: List[str] = []
    if np is None:
        missing.append("numpy")
    if Image is None or ImageDraw is None:
        missing.append("Pillow")
    if missing:
        raise ImportError("Missing dependencies: " + ", ".join(missing))


def import_sahi() -> Tuple[Any, Any, Any]:
    try:
        from sahi.models.base import DetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.prediction import ObjectPrediction
    except Exception as exc:
        raise ImportError(
            "SAHI is required for this script. Install it in the active environment, e.g. "
            "`python -m pip install sahi`."
        ) from exc
    return DetectionModel, get_sliced_prediction, ObjectPrediction


def parse_sample_selection(raw: str, default_min: int) -> Tuple[int, Optional[int], str]:
    txt = (raw or "").strip().lower()
    if not txt or txt == "all":
        return int(default_min), None, "all"

    if txt.endswith("+"):
        start = int(txt[:-1].strip())
        return start, None, txt

    if "-" in txt:
        left, right = txt.split("-", 1)
        start = int(left.strip())
        end = int(right.strip())
        if end < start:
            raise ValueError("Sample selection end must be >= start.")
        return start, end, txt

    value = int(txt)
    return value, value, txt


def normalize_class_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").strip().lower())


def parse_class_score_thresholds(
    raw: str,
    class_names: Sequence[str],
    default_threshold: float,
) -> Dict[str, float]:
    resolved = {str(name): float(default_threshold) for name in class_names}
    txt = (raw or "").strip()
    if not txt:
        return resolved

    known_by_key = {normalize_class_key(name): str(name) for name in class_names}
    parts = [part.strip() for part in txt.split(";") if part.strip()]
    for part in parts:
        if "=" not in part:
            raise ValueError(
                "Class score thresholds must use 'Class=0.50;OtherClass=0.70' format."
            )
        raw_key, raw_value = part.split("=", 1)
        key = normalize_class_key(raw_key)
        if key not in known_by_key:
            raise ValueError(
                f"Unknown class in class-score-thresholds: {raw_key!r}. "
                f"Known classes: {', '.join(class_names)}"
            )
        resolved[known_by_key[key]] = float(raw_value.strip())
    return resolved


def default_class_score_threshold_string() -> str:
    if not CLASS_SCORE_THRESHOLDS:
        return ""
    parts = [f"{name}={value}" for name, value in CLASS_SCORE_THRESHOLDS.items()]
    return ";".join(parts)


def threshold_for_class_idx(
    class_idx: int,
    class_names: Sequence[str],
    class_score_thresholds: Dict[str, float],
    default_threshold: float,
) -> float:
    if 0 <= int(class_idx) < len(class_names):
        return float(class_score_thresholds.get(class_names[int(class_idx)], default_threshold))
    return float(default_threshold)


def per_class_keep_mask(
    pred_scores: np.ndarray,
    pred_cls: np.ndarray,
    class_names: Sequence[str],
    class_score_thresholds: Dict[str, float],
    default_threshold: float,
) -> np.ndarray:
    if pred_scores.size == 0:
        return np.zeros((0,), dtype=bool)
    thresholds = np.array(
        [
            threshold_for_class_idx(int(cls_idx), class_names, class_score_thresholds, default_threshold)
            for cls_idx in pred_cls.tolist()
        ],
        dtype=np.float32,
    )
    return pred_scores >= thresholds


def discover_sample_dirs(images_root: Path, sample_min: int, sample_max: Optional[int] = None) -> List[Path]:
    dirs: List[Tuple[int, Path]] = []
    pattern = re.compile(r"^Sample\s+(\d+)$", re.IGNORECASE)
    for child in images_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name.strip())
        if not match:
            continue
        sample_idx = int(match.group(1))
        if sample_idx < int(sample_min):
            continue
        if sample_max is not None and sample_idx > int(sample_max):
            continue
        dirs.append((sample_idx, child))
    dirs.sort(key=lambda item: item[0])
    return [path for _, path in dirs]


def discover_images(sample_dirs: Sequence[Path]) -> List[Path]:
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    images: List[Path] = []
    for sample_dir in sample_dirs:
        direct_files = [
            path
            for path in sample_dir.iterdir()
            if path.is_file() and path.suffix.lower() in exts
        ]
        if direct_files:
            images.extend(sorted(direct_files, key=lambda p: str(p).lower()))
            continue
        for path in sample_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            if "patch" in str(path).lower():
                continue
            images.append(path)
    return sorted(images, key=lambda p: str(p).lower())


def extract_sample_name(image_path: Path) -> str:
    for part in image_path.parts[::-1]:
        if re.match(r"^Sample\s+\d+$", part, flags=re.IGNORECASE):
            return part
    return image_path.parent.name


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return path.name


def normalize_model_class_ids(pred_labels: np.ndarray, class_names: Sequence[str]) -> np.ndarray:
    if pred_labels.size == 0:
        return pred_labels.astype(np.int64)

    pred_labels = pred_labels.astype(np.int64)
    n_classes = len(class_names)
    if np.all((pred_labels >= 0) & (pred_labels < n_classes)):
        return pred_labels
    if np.all((pred_labels >= 1) & (pred_labels <= n_classes)):
        return pred_labels - 1
    clipped = np.clip(pred_labels, 0, max(0, n_classes - 1))
    return clipped.astype(np.int64)


def sanitize_boxes_xyxy(
    boxes: np.ndarray,
    image_w: int,
    image_h: int,
    min_size: float = 1e-3,
) -> np.ndarray:
    if boxes.size == 0:
        return boxes.astype(np.float32).reshape((0, 4))

    boxes = boxes.astype(np.float32).copy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, float(image_w))
    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, float(image_h))
    boxes[:, 2] = np.clip(boxes[:, 2], 0.0, float(image_w))
    boxes[:, 3] = np.clip(boxes[:, 3], 0.0, float(image_h))

    x1 = np.minimum(boxes[:, 0], boxes[:, 2])
    y1 = np.minimum(boxes[:, 1], boxes[:, 3])
    x2 = np.maximum(boxes[:, 0], boxes[:, 2])
    y2 = np.maximum(boxes[:, 1], boxes[:, 3])

    x2 = np.maximum(x2, x1 + float(min_size))
    y2 = np.maximum(y2, y1 + float(min_size))

    x2 = np.clip(x2, 0.0, float(image_w))
    y2 = np.clip(y2, 0.0, float(image_h))

    sanitized = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    keep = (sanitized[:, 2] > sanitized[:, 0]) & (sanitized[:, 3] > sanitized[:, 1])
    return sanitized[keep]


def score_to_text(score: float) -> str:
    return f"prob:{float(score):.2f}"


def _measure_text(draw: Any, text: str, font: Any) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return int(right - left), int(bottom - top)
    return draw.textsize(text, font=font)


def draw_text_with_outline(draw: Any, xy: Tuple[int, int], text: str, fill: Tuple[int, int, int], font: Any) -> None:
    x, y = xy
    outline = (0, 0, 0)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, fill=outline, font=font)
    draw.text((x, y), text, fill=fill, font=font)


def _score_anchor(box: Sequence[float], image_w: int, image_h: int, text_w: int, text_h: int) -> Tuple[int, int]:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    candidates = [
        (x1 + 2, max(0, y1 - text_h - 2)),
        (x1 + 2, min(image_h - text_h, y2 + 2)),
        (max(0, x2 - text_w - 2), max(0, y1 - text_h - 2)),
        (max(0, x2 - text_w - 2), min(image_h - text_h, y2 + 2)),
    ]
    for x, y in candidates:
        if 0 <= x <= max(0, image_w - text_w) and 0 <= y <= max(0, image_h - text_h):
            return x, y
    return max(0, min(x1, image_w - text_w)), max(0, min(y1, image_h - text_h))


def class_color_map(class_names: Sequence[str]) -> Dict[int, Tuple[int, int, int]]:
    palette = [
        (220, 30, 30),
        (20, 170, 80),
        (255, 140, 0),
        (40, 120, 220),
        (180, 80, 200),
        (0, 180, 180),
    ]
    return {i: palette[i % len(palette)] for i in range(len(class_names))}


def draw_prediction_overlay(
    image_path: Path,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_cls: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    colors = class_color_map(class_names)

    for box, score, cls_idx in zip(pred_boxes, pred_scores, pred_cls):
        cls_idx = int(cls_idx)
        color = colors.get(cls_idx, (255, 0, 0))
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = score_to_text(float(score))
        text_w, text_h = _measure_text(draw, text, font)
        tx, ty = _score_anchor(box.tolist(), img.width, img.height, text_w, text_h)
        draw_text_with_outline(draw, (tx, ty), text, fill=color, font=font)

    legend_lines = [f"{class_names[i]} = prediction" for i in range(len(class_names))]
    pad = 6
    line_h = max(_measure_text(draw, "Ag", font)[1], 10) + 2
    legend_w = max((_measure_text(draw, line, font)[0] for line in legend_lines), default=0)
    legend_h = len(legend_lines) * line_h + pad * 2
    draw.rectangle([6, 6, 6 + legend_w + pad * 2 + 18, 6 + legend_h], fill=(0, 0, 0))
    for idx, line in enumerate(legend_lines):
        color = colors.get(idx, (255, 0, 0))
        y = 6 + pad + idx * line_h
        draw.rectangle([12, y + 2, 24, y + line_h - 2], outline=color, width=2)
        draw_text_with_outline(draw, (30, y), line, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def build_sahi_model(model: Any, class_names: Sequence[str], confidence_threshold: float) -> Any:
    DetectionModel, _, ObjectPrediction = import_sahi()

    class RFDETRLocalSahiModel(DetectionModel):  # type: ignore[misc]
        required_packages: List[str] = []

        def set_model(self, model: Any, **kwargs) -> None:
            self.model = model

        def load_model(self) -> None:
            raise RuntimeError("Use an already loaded RF-DETR model instance with this wrapper.")

        def perform_inference(self, image: np.ndarray) -> None:
            img_pil = Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
            boxes, scores, labels = local_eval.predict_one_image(
                self.model,
                img_pil,
                float(self.confidence_threshold),
            )
            labels = normalize_model_class_ids(labels, class_names)
            raw_boxes = boxes.astype(np.float32).reshape((-1, 4)) if boxes.size else np.zeros((0, 4), dtype=np.float32)
            sanitized_boxes = sanitize_boxes_xyxy(raw_boxes, img_pil.width, img_pil.height)
            if len(sanitized_boxes) != len(raw_boxes):
                keep = []
                for box in raw_boxes:
                    clipped = sanitize_boxes_xyxy(box.reshape(1, 4), img_pil.width, img_pil.height)
                    keep.append(len(clipped) == 1)
                keep_mask = np.asarray(keep, dtype=bool)
                scores = scores[keep_mask]
                labels = labels[keep_mask]
            boxes = sanitized_boxes
            self._original_predictions = [(boxes, scores, labels)]

        def _create_object_prediction_list_from_original_predictions(
            self,
            shift_amount_list: Optional[List[List[int]]] = None,
            full_shape_list: Optional[List[List[int]]] = None,
        ) -> None:
            try:
                from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
            except Exception as exc:
                raise ImportError("SAHI compatibility helpers are unavailable.") from exc

            shift_amount_list = fix_shift_amount_list(shift_amount_list)
            full_shape_list = fix_full_shape_list(full_shape_list)
            object_prediction_list: List[Any] = []
            predictions = self._original_predictions or []
            if len(predictions) != len(shift_amount_list) or len(predictions) != len(full_shape_list):
                raise ValueError("Length mismatch between predictions, shifts, and full shapes.")
            for (boxes, scores, labels), shift_amount, full_shape in zip(
                predictions,
                shift_amount_list,
                full_shape_list,
            ):
                for box, score, label in zip(boxes, scores, labels):
                    cls_idx = int(label)
                    cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else f"class_{cls_idx}"
                    object_prediction_list.append(
                        ObjectPrediction(
                            bbox=[float(v) for v in box.tolist()],
                            category_id=cls_idx,
                            category_name=cls_name,
                            score=float(score),
                            shift_amount=shift_amount,
                            full_shape=full_shape,
                        )
                    )
            self._object_prediction_list_per_image = [object_prediction_list]

    category_mapping = {str(i): name for i, name in enumerate(class_names)}
    return RFDETRLocalSahiModel(
        model=model,
        confidence_threshold=float(confidence_threshold),
        category_mapping=category_mapping,
        load_at_init=True,
    )


def discover_gt_image_mapping(coco: Any) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    for img_id in coco.getImgIds():
        image_info = coco.loadImgs([img_id])[0]
        file_name = str(image_info.get("file_name", "")).replace("\\", "/")
        mapping[file_name] = image_info
        mapping[Path(file_name).name] = image_info
    return mapping


def gt_category_maps(coco: Any) -> Tuple[Dict[int, str], Dict[str, int]]:
    cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])
    id_to_name = {int(cat["id"]): str(cat["name"]) for cat in cats}
    name_to_id = {str(cat["name"]).strip().lower(): int(cat["id"]) for cat in cats}
    return id_to_name, name_to_id


def gt_cat_id_for_name(class_name: str, gt_name_to_id: Dict[str, int]) -> int:
    key = class_name.strip().lower()
    if key in gt_name_to_id:
        return gt_name_to_id[key]
    if class_name == "Squamous Epithelial Cell" and "epithelial" in gt_name_to_id:
        return gt_name_to_id["epithelial"]
    if class_name == "Leucocyte" and "leucocyte" in gt_name_to_id:
        return gt_name_to_id["leucocyte"]
    raise KeyError(f"Could not map class {class_name!r} to a GT COCO category.")


def detection_counts_by_class(
    samples: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    score_threshold: float,
    class_score_thresholds: Dict[str, float],
    iou_thr: float,
) -> List[Dict[str, int]]:
    rows: List[Dict[str, int]] = []
    for class_idx in range(len(class_names)):
        tp = fp = fn = 0
        for sample in samples:
            gt_mask = sample["gt_cls"] == class_idx
            pr_mask = sample["pred_cls"] == class_idx
            gt_boxes = sample["gt_boxes"][gt_mask]
            pred_boxes = sample["pred_boxes"][pr_mask]
            pred_scores = sample["pred_scores"][pr_mask]
            class_threshold = threshold_for_class_idx(
                class_idx,
                class_names,
                class_score_thresholds,
                score_threshold,
            )
            keep = pred_scores >= class_threshold
            pred_boxes = pred_boxes[keep]

            ious = local_eval.iou_matrix(gt_boxes, pred_boxes)
            matches = local_eval.greedy_match(ious, iou_thr=iou_thr)
            tp += len(matches)
            fp += max(0, len(pred_boxes) - len(matches))
            fn += max(0, len(gt_boxes) - len(matches))
        rows.append({"tp": tp, "fp": fp, "fn": fn})
    return rows


def confusion_matrix_with_background_per_class(
    samples: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    score_threshold: float,
    class_score_thresholds: Dict[str, float],
    iou_thr: float,
) -> np.ndarray:
    n_classes = len(class_names)
    bg = n_classes
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
    for sample in samples:
        gt_boxes = sample["gt_boxes"]
        gt_cls = sample["gt_cls"]

        keep = per_class_keep_mask(
            sample["pred_scores"],
            sample["pred_cls"],
            class_names,
            class_score_thresholds,
            score_threshold,
        )
        pred_boxes = sample["pred_boxes"][keep]
        pred_cls = sample["pred_cls"][keep]

        ious = local_eval.iou_matrix(gt_boxes, pred_boxes)
        matches = local_eval.greedy_match(ious, iou_thr=iou_thr)
        matched_gt = {gi for gi, _ in matches}
        matched_pr = {pj for _, pj in matches}

        for gi, pj in matches:
            cm[int(gt_cls[gi]), int(pred_cls[pj])] += 1
        for gi in range(len(gt_cls)):
            if gi not in matched_gt:
                cm[int(gt_cls[gi]), bg] += 1
        for pj in range(len(pred_cls)):
            if pj not in matched_pr:
                cm[bg, int(pred_cls[pj])] += 1
    return cm


def precision_recall_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = float(tp / max(1, tp + fp))
    recall = float(tp / max(1, tp + fn))
    f1 = float((2.0 * precision * recall) / max(1e-9, precision + recall))
    return precision, recall, f1


def summarize_coco_metrics(
    coco: Any,
    preds_path: Path,
    processed_img_ids: Sequence[int],
    samples: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    score_threshold: float,
    class_score_thresholds: Dict[str, float],
    confmat_iou: float,
    output_dir: Path,
    no_plots: bool,
) -> Dict[str, Any]:
    cat_id_to_name, id_to_idx, idx_to_id = local_eval.coco_categories(coco)
    std_iou = np.linspace(0.50, 0.95, 10, dtype=np.float64)
    iou_sweep = np.round(np.arange(0.50, 0.95 + 0.05, 0.05), 2)

    coco_std = local_eval.run_coco_eval(coco, preds_path, processed_img_ids, std_iou)
    coco_std.summarize()
    coco_sweep = local_eval.run_coco_eval(coco, preds_path, processed_img_ids, iou_sweep)
    precision = coco_sweep.eval["precision"]
    recall = coco_sweep.eval["recall"]
    cat_ids_eval = list(coco_sweep.params.catIds)
    idx50 = local_eval.find_iou_index(iou_sweep, 0.50)
    idx75 = local_eval.find_iou_index(iou_sweep, 0.75)

    per_class_rows: List[Dict[str, Any]] = []
    prf_counts = detection_counts_by_class(
        samples,
        class_names,
        score_threshold,
        class_score_thresholds,
        confmat_iou,
    )
    internal_idx_by_name = {name.strip().lower(): idx for idx, name in enumerate(class_names)}
    for k, cid in enumerate(cat_ids_eval):
        class_name = cat_id_to_name[int(cid)]
        count_idx = internal_idx_by_name.get(class_name.strip().lower(), k)
        counts = prf_counts[count_idx] if count_idx < len(prf_counts) else {"tp": 0, "fp": 0, "fn": 0}
        class_threshold = threshold_for_class_idx(
            count_idx,
            class_names,
            class_score_thresholds,
            score_threshold,
        )
        prec_thr, rec_thr, f1_thr = precision_recall_from_counts(
            counts["tp"],
            counts["fp"],
            counts["fn"],
        )
        per_class_rows.append(
            {
                "category_id": int(cid),
                "class": class_name,
                "AP@50:95": local_eval.ap_from_precision_slice(precision[:, :, k, 0, -1]),
                "AP@50": local_eval.ap_from_precision_slice(precision[idx50, :, k, 0, -1]) if idx50 is not None else float("nan"),
                "AP@75": local_eval.ap_from_precision_slice(precision[idx75, :, k, 0, -1]) if idx75 is not None else float("nan"),
                "AR@50:95": local_eval.ar_from_recall_slice(recall[:, k, 0, -1]),
                "AR@50": local_eval.ar_from_recall_slice(recall[idx50, k, 0, -1]) if idx50 is not None else float("nan"),
                "precision": prec_thr,
                "recall": rec_thr,
                "f1": f1_thr,
                "score_threshold": class_threshold,
                "tp": counts["tp"],
                "fp": counts["fp"],
                "fn": counts["fn"],
            }
        )

    fieldnames = [
        "category_id",
        "class",
        "AP@50:95",
        "AP@50",
        "AP@75",
        "AR@50:95",
        "AR@50",
        "precision",
        "recall",
        "f1",
        "score_threshold",
        "tp",
        "fp",
        "fn",
    ]
    write_csv(output_dir / "per_class_metrics.csv", fieldnames, per_class_rows)

    cm = confusion_matrix_with_background_per_class(
        samples,
        class_names,
        score_threshold,
        class_score_thresholds,
        confmat_iou,
    )
    cm_labels = list(class_names) + ["background"]
    local_eval.save_confusion_csv(output_dir / "confusion_matrix.csv", cm_labels, cm)
    json_dump(
        output_dir / "confusion_matrix.json",
        {
            "labels": cm_labels,
            "matrix": cm.tolist(),
            "score_threshold": score_threshold,
            "class_score_thresholds": class_score_thresholds,
            "iou_threshold": confmat_iou,
        },
    )
    confusion_png_created = False
    if not no_plots:
        confusion_png_created = local_eval.save_confusion_matrix_plot(
            output_dir / "confusion_matrix.png",
            cm_labels,
            cm,
            score_threshold,
            confmat_iou,
        )

    overall_tp = int(sum(row["tp"] for row in prf_counts))
    overall_fp = int(sum(row["fp"] for row in prf_counts))
    overall_fn = int(sum(row["fn"] for row in prf_counts))
    overall_precision, overall_recall, overall_f1 = precision_recall_from_counts(
        overall_tp,
        overall_fp,
        overall_fn,
    )
    return {
        "map@50:95": float(coco_std.stats[0]),
        "map@50": float(coco_std.stats[1]),
        "map@75": float(coco_std.stats[2]),
        "mar@50:95": float(coco_std.stats[8]),
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "score_threshold": score_threshold,
        "class_score_thresholds": class_score_thresholds,
        "per_class": per_class_rows,
        "confusion_matrix_png_created": bool(confusion_png_created),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local SAHI full-FOV RF-DETR inference/evaluation.")
    parser.add_argument("--preset", default=ACTIVE_PRESET, choices=sorted(FOV_PRESETS))
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--images-root", default="")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gt-json", default="")
    parser.add_argument("--target-name", default="")
    parser.add_argument("--model-class", default="auto", choices=["auto", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"])
    parser.add_argument("--model-resolution", type=int, default=None)
    parser.add_argument("--sample-min", type=int, default=DEFAULT_SAMPLE_MIN)
    parser.add_argument("--sample-selection", default=SAMPLE_SELECTION)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--class-score-thresholds", default=default_class_score_threshold_string())
    parser.add_argument("--confmat-iou", type=float, default=0.50)
    parser.add_argument("--slice-height", type=int, default=672)
    parser.add_argument("--slice-width", type=int, default=672)
    parser.add_argument("--overlap-height-ratio", type=float, default=0.20)
    parser.add_argument("--overlap-width-ratio", type=float, default=0.20)
    parser.add_argument("--perform-standard-pred", action="store_true", default=True)
    parser.add_argument("--no-standard-pred", dest="perform_standard_pred", action="store_false")
    parser.add_argument("--postprocess-type", default="GREEDYNMM", choices=["GREEDYNMM", "NMM", "NMS", "LSNMS"])
    parser.add_argument("--postprocess-match-metric", default="IOU", choices=["IOU", "IOS"])
    parser.add_argument("--postprocess-match-threshold", type=float, default=0.50)
    parser.add_argument("--postprocess-class-agnostic", action="store_true")
    parser.add_argument("--overlay-selection", default=OVERLAY_SELECTION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def resolve_config(args: argparse.Namespace) -> FOVSahiConfig:
    preset = _resolve_preset(args.preset)
    checkpoint = _optional_path(args.checkpoint) or Path(preset["checkpoint"])
    images_root = _optional_path(args.images_root) or Path(preset["images_root"])
    base_output_dir = Path(args.output_dir).expanduser()
    target_name = (args.target_name or "").strip() or str(preset.get("target_name", "TwoClass"))
    output_dir, output_root_dir, run_timestamp = build_output_dir(base_output_dir, target_name)
    num_overlays = parse_overlay_selection(args.overlay_selection)
    class_names = list(preset.get("class_names", DEFAULT_CLASS_NAMES))
    gt_json = _optional_path(args.gt_json)
    sample_min, sample_max, sample_selection = parse_sample_selection(
        str(args.sample_selection),
        int(args.sample_min),
    )
    class_score_thresholds = parse_class_score_thresholds(
        str(args.class_score_thresholds),
        class_names,
        float(args.score_threshold),
    )

    model_class = args.model_class
    if model_class == "auto":
        model_class = local_eval.infer_model_class(checkpoint.parent, checkpoint)
    model_resolution = args.model_resolution
    if model_resolution is None:
        model_resolution = local_eval.infer_model_resolution(checkpoint)

    return FOVSahiConfig(
        preset_name=args.preset,
        checkpoint=checkpoint,
        images_root=images_root,
        output_dir=output_dir,
        output_root_dir=output_root_dir,
        run_timestamp=run_timestamp,
        target_name=target_name,
        class_names=class_names,
        gt_json=gt_json,
        model_class=model_class,
        model_resolution=model_resolution,
        sample_min=sample_min,
        sample_max=sample_max,
        sample_selection=sample_selection,
        score_floor=float(args.score_floor),
        score_threshold=float(args.score_threshold),
        class_score_thresholds=class_score_thresholds,
        confmat_iou=float(args.confmat_iou),
        seed=int(args.seed),
        progress_every=max(1, int(args.progress_every)),
        num_overlays=num_overlays,
        no_plots=bool(args.no_plots),
        slice_height=int(args.slice_height) if args.slice_height else None,
        slice_width=int(args.slice_width) if args.slice_width else None,
        overlap_height_ratio=float(args.overlap_height_ratio),
        overlap_width_ratio=float(args.overlap_width_ratio),
        perform_standard_pred=bool(args.perform_standard_pred),
        postprocess_type=str(args.postprocess_type),
        postprocess_match_metric=str(args.postprocess_match_metric),
        postprocess_match_threshold=float(args.postprocess_match_threshold),
        postprocess_class_agnostic=bool(args.postprocess_class_agnostic),
    )


def run_fov_sahi_eval(cfg: FOVSahiConfig) -> None:
    ensure_deps()
    _, get_sliced_prediction, _ = import_sahi()

    if not cfg.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint}")
    if not cfg.images_root.exists():
        raise FileNotFoundError(f"Images root not found: {cfg.images_root}")
    if cfg.gt_json is not None and not cfg.gt_json.exists():
        raise FileNotFoundError(f"GT JSON not found: {cfg.gt_json}")

    sample_dirs = discover_sample_dirs(cfg.images_root, cfg.sample_min, cfg.sample_max)
    image_paths = discover_images(sample_dirs)
    if not image_paths:
        raise RuntimeError(
            f"No images found under {cfg.images_root} for sample selection {cfg.sample_selection!r}."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = cfg.output_dir / "overlays"
    random.seed(cfg.seed)

    print(f"[SAHI FOV] Preset={cfg.preset_name}")
    print(f"[SAHI FOV] Target={cfg.target_name}")
    print(f"[SAHI FOV] Checkpoint={cfg.checkpoint}")
    print(f"[SAHI FOV] Images root={cfg.images_root}")
    print(f"[SAHI FOV] Sample selection={cfg.sample_selection}")
    print(f"[SAHI FOV] Sample min={cfg.sample_min}")
    print(f"[SAHI FOV] Sample max={cfg.sample_max}")
    print(f"[SAHI FOV] Discovered sample dirs={len(sample_dirs)}")
    print(f"[SAHI FOV] Discovered images={len(image_paths)}")
    print(f"[SAHI FOV] Output directory={cfg.output_dir}")
    if cfg.gt_json is not None:
        print(f"[SAHI FOV] GT JSON={cfg.gt_json}")

    ckpt_num_classes, ckpt_class_names = infer_checkpoint_runtime_metadata(cfg.checkpoint)
    effective_class_names = list(cfg.class_names)
    if ckpt_class_names:
        effective_class_names = list(ckpt_class_names)
    effective_num_classes = ckpt_num_classes if ckpt_num_classes is not None else len(effective_class_names)

    print(f"[SAHI FOV] Checkpoint num_classes={effective_num_classes}")
    print(f"[SAHI FOV] Checkpoint class_names={effective_class_names}")
    print(f"[SAHI FOV] Score threshold={cfg.score_threshold}")

    model = load_model_for_fov(
        cfg.model_class,
        cfg.checkpoint,
        cfg.model_resolution,
        effective_num_classes,
        effective_class_names,
    )
    sahi_model = build_sahi_model(model, effective_class_names, cfg.score_floor)
    effective_class_score_thresholds = parse_class_score_thresholds(
        ";".join(f"{name}={cfg.class_score_thresholds.get(name, cfg.score_threshold)}" for name in effective_class_names),
        effective_class_names,
        cfg.score_threshold,
    )
    print(f"[SAHI FOV] Class score thresholds={effective_class_score_thresholds}")

    gt_coco = None
    gt_image_lookup: Dict[str, Dict[str, Any]] = {}
    gt_name_to_id: Dict[str, int] = {}
    gt_cat_id_to_internal: Dict[int, int] = {}
    if cfg.gt_json is not None:
        gt_coco = local_eval.COCO(str(cfg.gt_json))
        gt_image_lookup = discover_gt_image_mapping(gt_coco)
        _, gt_name_to_id = gt_category_maps(gt_coco)
        for idx, class_name in enumerate(effective_class_names):
            gt_cat_id_to_internal[gt_cat_id_for_name(class_name, gt_name_to_id)] = idx

    overlay_candidates = list(range(len(image_paths)))
    if cfg.num_overlays == 0:
        overlay_indices: set[int] = set()
    elif cfg.num_overlays < 0 or cfg.num_overlays >= len(image_paths):
        overlay_indices = set(overlay_candidates)
    else:
        overlay_indices = set(random.sample(overlay_candidates, cfg.num_overlays))

    per_image_rows: List[Dict[str, Any]] = []
    per_sample_accum: Dict[str, Dict[str, Any]] = {}
    all_prediction_rows: List[Dict[str, Any]] = []
    coco_predictions: List[Dict[str, Any]] = []
    processed_img_ids: List[int] = []
    metric_samples: List[Dict[str, Any]] = []
    unmatched_gt_images: List[str] = []

    for idx, image_path in enumerate(image_paths, start=1):
        if idx == 1 or (idx % cfg.progress_every) == 0 or idx == len(image_paths):
            pct = 100.0 * idx / max(1, len(image_paths))
            print(f"[SAHI FOV] Inference progress: {idx}/{len(image_paths)} ({pct:.1f}%)")

        prediction_result = get_sliced_prediction(
            str(image_path),
            detection_model=sahi_model,
            slice_height=cfg.slice_height,
            slice_width=cfg.slice_width,
            overlap_height_ratio=cfg.overlap_height_ratio,
            overlap_width_ratio=cfg.overlap_width_ratio,
            perform_standard_pred=cfg.perform_standard_pred,
            postprocess_type=cfg.postprocess_type,
            postprocess_match_metric=cfg.postprocess_match_metric,
            postprocess_match_threshold=cfg.postprocess_match_threshold,
            postprocess_class_agnostic=cfg.postprocess_class_agnostic,
            verbose=0,
        )

        pred_boxes_list: List[List[float]] = []
        pred_scores_list: List[float] = []
        pred_cls_list: List[int] = []
        for obj in prediction_result.object_prediction_list:
            x1, y1, x2, y2 = obj.bbox.to_xyxy()
            pred_boxes_list.append([float(x1), float(y1), float(x2), float(y2)])
            pred_scores_list.append(float(obj.score.value))
            pred_cls_list.append(int(obj.category.id))

        pred_boxes = np.asarray(pred_boxes_list, dtype=np.float32).reshape((-1, 4)) if pred_boxes_list else np.zeros((0, 4), dtype=np.float32)
        pred_scores = np.asarray(pred_scores_list, dtype=np.float32) if pred_scores_list else np.zeros((0,), dtype=np.float32)
        pred_cls = np.asarray(pred_cls_list, dtype=np.int64) if pred_cls_list else np.zeros((0,), dtype=np.int64)
        pred_cls = normalize_model_class_ids(pred_cls, effective_class_names)
        keep = per_class_keep_mask(
            pred_scores,
            pred_cls,
            effective_class_names,
            effective_class_score_thresholds,
            cfg.score_threshold,
        )
        kept_boxes = pred_boxes[keep]
        kept_scores = pred_scores[keep]
        kept_cls = pred_cls[keep]

        sample_name = extract_sample_name(image_path)
        rel_name = safe_relpath(image_path, cfg.images_root)
        counts_by_class = {f"n_pred_{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_').lower()}": 0 for name in effective_class_names}
        for cls_idx in kept_cls.tolist():
            key = f"n_pred_{re.sub(r'[^A-Za-z0-9]+', '_', effective_class_names[int(cls_idx)]).strip('_').lower()}"
            counts_by_class[key] += 1

        image_row = {
            "sample": sample_name,
            "file_name": rel_name,
            "image_name": image_path.name,
            "n_predictions_raw": int(len(pred_boxes)),
            "n_predictions_kept": int(len(kept_boxes)),
        }
        image_row.update(counts_by_class)
        per_image_rows.append(image_row)

        sample_row = per_sample_accum.setdefault(
            sample_name,
            {
                "sample": sample_name,
                "n_images": 0,
                "n_predictions_raw": 0,
                "n_predictions_kept": 0,
                **{key: 0 for key in counts_by_class},
            },
        )
        sample_row["n_images"] += 1
        sample_row["n_predictions_raw"] += int(len(pred_boxes))
        sample_row["n_predictions_kept"] += int(len(kept_boxes))
        for key, value in counts_by_class.items():
            sample_row[key] += int(value)

        for box, score, cls_idx in zip(pred_boxes, pred_scores, pred_cls):
            cls_idx = int(cls_idx)
            cls_name = effective_class_names[cls_idx] if 0 <= cls_idx < len(effective_class_names) else f"class_{cls_idx}"
            all_prediction_rows.append(
                {
                    "sample": sample_name,
                    "file_name": rel_name,
                    "image_name": image_path.name,
                    "bbox_xyxy": [float(v) for v in box.tolist()],
                    "score": float(score),
                    "class_index": cls_idx,
                    "class_name": cls_name,
                }
            )

        gt_info = None
        if gt_coco is not None:
            gt_info = gt_image_lookup.get(rel_name.replace("\\", "/")) or gt_image_lookup.get(image_path.name)
            if gt_info is None:
                unmatched_gt_images.append(rel_name)
            else:
                gt_image_id = int(gt_info["id"])
                processed_img_ids.append(gt_image_id)
                ann_ids = gt_coco.getAnnIds(imgIds=[gt_image_id], iscrowd=None)
                anns = gt_coco.loadAnns(ann_ids)
                gt_boxes_list: List[List[float]] = []
                gt_cls_list: List[int] = []
                for ann in anns:
                    cat_id = int(ann.get("category_id", -1))
                    if cat_id not in gt_cat_id_to_internal:
                        continue
                    x, y, w, h = ann.get("bbox", [0, 0, 0, 0])
                    gt_boxes_list.append([float(x), float(y), float(x + w), float(y + h)])
                    gt_cls_list.append(int(gt_cat_id_to_internal[cat_id]))
                gt_boxes = np.asarray(gt_boxes_list, dtype=np.float32).reshape((-1, 4)) if gt_boxes_list else np.zeros((0, 4), dtype=np.float32)
                gt_cls = np.asarray(gt_cls_list, dtype=np.int64) if gt_cls_list else np.zeros((0,), dtype=np.int64)
                metric_samples.append(
                    {
                        "file_name": rel_name,
                        "gt_boxes": gt_boxes,
                        "gt_cls": gt_cls,
                        "pred_boxes": pred_boxes,
                        "pred_scores": pred_scores,
                        "pred_cls": pred_cls,
                    }
                )
                for box, score, cls_idx in zip(pred_boxes, pred_scores, pred_cls):
                    class_name = effective_class_names[int(cls_idx)]
                    coco_predictions.append(
                        {
                            "image_id": gt_image_id,
                            "category_id": gt_cat_id_for_name(class_name, gt_name_to_id),
                            "bbox": local_eval.to_coco_bbox_xywh(box),
                            "score": float(score),
                        }
                    )

        if (idx - 1) in overlay_indices:
            draw_prediction_overlay(
                image_path=image_path,
                pred_boxes=kept_boxes,
                pred_scores=kept_scores,
                pred_cls=kept_cls,
                class_names=effective_class_names,
                output_path=overlays_dir / f"{sample_name}__{image_path.stem}.png",
            )

    per_sample_rows = sorted(per_sample_accum.values(), key=lambda row: row["sample"])
    per_image_fieldnames = list(per_image_rows[0].keys()) if per_image_rows else ["sample", "file_name", "image_name", "n_predictions_raw", "n_predictions_kept"]
    per_sample_fieldnames = list(per_sample_rows[0].keys()) if per_sample_rows else ["sample", "n_images", "n_predictions_raw", "n_predictions_kept"]
    write_csv(cfg.output_dir / "per_image_counts.csv", per_image_fieldnames, per_image_rows)
    write_csv(cfg.output_dir / "per_sample_counts.csv", per_sample_fieldnames, per_sample_rows)
    json_dump(cfg.output_dir / "predictions.json", all_prediction_rows)

    metrics_summary: Optional[Dict[str, Any]] = None
    if gt_coco is not None:
        preds_path = cfg.output_dir / "predictions_coco.json"
        json_dump(preds_path, coco_predictions)
        metrics_summary = summarize_coco_metrics(
            coco=gt_coco,
            preds_path=preds_path,
            processed_img_ids=processed_img_ids,
            samples=metric_samples,
            class_names=effective_class_names,
            score_threshold=cfg.score_threshold,
            class_score_thresholds=effective_class_score_thresholds,
            confmat_iou=cfg.confmat_iou,
            output_dir=cfg.output_dir,
            no_plots=cfg.no_plots,
        )

    summary = {
        "target_name": cfg.target_name,
        "preset_name": cfg.preset_name,
        "checkpoint": str(cfg.checkpoint),
        "images_root": str(cfg.images_root),
        "gt_json": str(cfg.gt_json) if cfg.gt_json is not None else None,
        "sample_selection": cfg.sample_selection,
        "sample_min": cfg.sample_min,
        "sample_max": cfg.sample_max,
        "n_samples": len(sample_dirs),
        "n_images": len(image_paths),
        "class_names": list(effective_class_names),
        "score_floor": cfg.score_floor,
        "score_threshold": cfg.score_threshold,
        "class_score_thresholds": effective_class_score_thresholds,
        "confmat_iou": cfg.confmat_iou,
        "slice_height": cfg.slice_height,
        "slice_width": cfg.slice_width,
        "overlap_height_ratio": cfg.overlap_height_ratio,
        "overlap_width_ratio": cfg.overlap_width_ratio,
        "perform_standard_pred": cfg.perform_standard_pred,
        "postprocess_type": cfg.postprocess_type,
        "postprocess_match_metric": cfg.postprocess_match_metric,
        "postprocess_match_threshold": cfg.postprocess_match_threshold,
        "postprocess_class_agnostic": cfg.postprocess_class_agnostic,
        "num_overlays_requested": cfg.num_overlays,
        "num_overlays_written": len(overlay_indices),
        "unmatched_gt_images": unmatched_gt_images[:100],
        "metrics": metrics_summary,
    }
    json_dump(cfg.output_dir / "fov_summary.json", summary)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = resolve_config(args)
    run_fov_sahi_eval(cfg)


if __name__ == "__main__":
    main()
