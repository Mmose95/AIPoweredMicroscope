#!/usr/bin/env python
"""
Local RF-DETR evaluation script.

This script is intentionally local-only:
- No /work or UCloud path handling.
- No remote path rewrite logic.
- Assumes model checkpoint + COCO test json + image files are available locally.

Outputs:
- eval_summary.json
- predictions_coco.json
- iou_sweep_metrics.csv
- per_class_iou_sweep.csv
- per_class_metrics.csv
- confusion_matrix.json
- confusion_matrix.csv
- threshold_metrics.csv
- pr_curve_overall.csv (if sklearn available)
- roc_curve_overall.csv (if sklearn available and both classes present)
- optional PNG plots + overlays/
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ============================================================================
# USER PATH INPUTS (EDIT THESE)
# ============================================================================
# Set explicit local paths here.
# You can also override each one from CLI:
#   --checkpoint ... --test-json ... --output-dir ... --images-root ...
CHECKPOINT = r"D:\PHD\Results\Quality Assessment\Epi+Leu for ESCMID Conference\first full Epi model no SSL\HPO_Config_003/checkpoint_best_total.pth"
TEST_JSON = r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\SOLO_Supervised_RFDETR\Stat_Dataset\QA-2025v2_SquamousEpithelialCell_OVR_20260217-093944\test/_annotations.coco.json"
OUTPUT_DIR = r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\EvaluationOutput"
# Prefix applied to relative file_name entries from COCO JSON, e.g.
# "Sample 15/Patches for Sample 15/....tif"
IMAGE_ROOT = r"D:\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment_Cleaned"


def _optional_path(raw: str) -> Optional[Path]:
    raw = (raw or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()

try:
    import numpy as np
except Exception:
    np = None  # type: ignore[assignment]

try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception:
    COCO = None  # type: ignore[assignment]
    COCOeval = None  # type: ignore[assignment]

try:
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
except Exception:
    auc = None  # type: ignore[assignment]
    average_precision_score = None  # type: ignore[assignment]
    precision_recall_curve = None  # type: ignore[assignment]
    roc_curve = None  # type: ignore[assignment]


@dataclass
class LocalEvalConfig:
    checkpoint: Path
    test_json: Path
    output_dir: Path
    images_root: Optional[Path]
    model_class: str
    score_floor: float
    score_threshold: float
    confmat_iou: float
    curve_iou: float
    iou_min: float
    iou_max: float
    iou_step: float
    threshold_points: int
    max_images: Optional[int]
    num_overlays: int
    image_max_side: int
    seed: int
    skip_missing_images: bool
    no_plots: bool


def ensure_deps() -> None:
    missing: List[str] = []
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if Image is None or ImageDraw is None:
        missing.append("Pillow")
    if COCO is None or COCOeval is None:
        missing.append("pycocotools")
    if missing:
        raise ImportError("Missing dependencies: " + ", ".join(missing))


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def infer_model_class(model_meta_root: Path, checkpoint: Path) -> str:
    meta_model = model_meta_root / "rfdetr_run" / "run_meta" / "model_architecture.json"
    if meta_model.exists():
        try:
            js = json.loads(meta_model.read_text(encoding="utf-8"))
            name = str(js.get("model_name", "")).strip()
            if name in {"RFDETRSmall", "RFDETRMedium", "RFDETRLarge"}:
                return name
        except Exception:
            pass

    ckpt_name = checkpoint.name.lower()
    if "small" in ckpt_name:
        return "RFDETRSmall"
    if "medium" in ckpt_name:
        return "RFDETRMedium"
    return "RFDETRLarge"


def load_model(model_class: str, checkpoint: Path):
    from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge

    name_to_cls = {
        "RFDETRSmall": RFDETRSmall,
        "RFDETRMedium": RFDETRMedium,
        "RFDETRLarge": RFDETRLarge,
    }
    if model_class not in name_to_cls:
        raise ValueError(f"Unsupported model_class={model_class}")

    model = name_to_cls[model_class](pretrain_weights=str(checkpoint))
    if hasattr(model, "optimize_for_inference"):
        try:
            model.optimize_for_inference()
        except Exception:
            pass
    return model


def resolve_image_path_local(file_name: str, test_json: Path, images_root: Optional[Path]) -> Path:
    candidates: List[Path] = []
    raw = file_name or ""
    p_raw = Path(raw)
    if p_raw.is_absolute():
        candidates.append(p_raw)

    rel = raw.replace("\\", "/")
    candidates.append(test_json.parent / rel)
    candidates.append(test_json.parent / Path(rel).name)
    if images_root is not None and rel:
        candidates.append(images_root / rel)
        candidates.append(images_root / Path(rel).name)

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(
        f"Could not resolve local image path for '{file_name}'. "
        f"Checked absolute + relative to {test_json.parent}."
    )


def predict_one_image(model: Any, img_pil: Image.Image, score_floor: float):
    arr = np.array(img_pil.convert("RGB"))
    ten = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    out = None
    for name in ("predict", "infer", "inference", "forward_inference"):
        fn = getattr(model, name, None)
        if callable(fn):
            for inp in (img_pil, arr, ten):
                try:
                    out = fn(inp)
                    break
                except Exception:
                    out = None
            if out is not None:
                break

    if out is None:
        forward = getattr(model, "forward", None)
        if callable(forward):
            out = forward(ten)
        else:
            raise RuntimeError("Model has no supported inference method.")

    try:
        import supervision as sv

        if isinstance(out, sv.Detections):
            boxes = out.xyxy.astype(np.float32)
            scores = out.confidence.astype(np.float32) if getattr(out, "confidence", None) is not None else np.ones((len(boxes),), dtype=np.float32)
            labels = out.class_id.astype(np.int64) if getattr(out, "class_id", None) is not None else np.zeros((len(boxes),), dtype=np.int64)
            keep = scores >= score_floor
            return boxes[keep], scores[keep], labels[keep]
    except Exception:
        pass

    if isinstance(out, (list, tuple)) and len(out) == 1:
        out = out[0]
    if isinstance(out, (list, tuple)) and len(out) == 3:
        b, s, l = out
        out = {"boxes": b, "scores": s, "labels": l}

    if not isinstance(out, dict):
        raise RuntimeError(f"Prediction output type not recognized: {type(out)}")

    key_sets = [
        ("boxes", "scores", "labels"),
        ("pred_boxes", "scores", "labels"),
        ("bboxes", "scores", "classes"),
        ("boxes_xyxy", "scores", "labels"),
        ("detections", None, None),
    ]
    boxes = scores = labels = None
    for kb, ks, kl in key_sets:
        if kb in out and (ks is None or ks in out) and (kl is None or kl in out):
            if kb == "detections":
                d = out["detections"]
                boxes = d.get("boxes") or d.get("bboxes")
                scores = d.get("scores")
                labels = d.get("labels") or d.get("classes")
            else:
                boxes = out[kb]
                scores = out.get(ks)
                labels = out.get(kl)
            break

    if boxes is None:
        raise RuntimeError(f"Unrecognized prediction dict keys: {list(out.keys())}")

    def to_np(x: Any) -> np.ndarray:
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    boxes_np = to_np(boxes).astype(np.float32)
    scores_np = to_np(scores).astype(np.float32) if scores is not None else np.ones((len(boxes_np),), dtype=np.float32)
    labels_np = to_np(labels).astype(np.int64) if labels is not None else np.zeros((len(boxes_np),), dtype=np.int64)

    keep = scores_np >= score_floor
    return boxes_np[keep], scores_np[keep], labels_np[keep]


def coco_categories(coco: COCO) -> Tuple[Dict[int, str], Dict[int, int], Dict[int, int]]:
    cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    ordered_ids = [c["id"] for c in cats]
    id_to_idx = {cid: i for i, cid in enumerate(ordered_ids)}
    idx_to_id = {i: cid for cid, i in id_to_idx.items()}
    return cat_id_to_name, id_to_idx, idx_to_id


def to_coco_cat_ids(pred_labels: np.ndarray, cat_id_to_name: Dict[int, str], n_classes: int, idx_to_id: Dict[int, int]) -> np.ndarray:
    pred_labels = pred_labels.astype(int)
    cat_ids = set(cat_id_to_name.keys())

    if len(pred_labels) > 0 and np.all(np.isin(pred_labels, list(cat_ids))):
        return pred_labels.astype(np.int64)
    if np.all((pred_labels >= 0) & (pred_labels < n_classes)):
        return np.array([idx_to_id[int(i)] for i in pred_labels], dtype=np.int64)

    clipped = np.clip(pred_labels, 0, n_classes - 1)
    return np.array([idx_to_id[int(i)] for i in clipped], dtype=np.int64)


def to_coco_bbox_xywh(box_xyxy: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = box_xyxy.tolist()
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)

    inter_x1 = np.maximum(x11, x21.T)
    inter_y1 = np.maximum(y11, y21.T)
    inter_x2 = np.minimum(x12, x22.T)
    inter_y2 = np.minimum(y12, y22.T)

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area1 = np.clip(x12 - x11, 0, None) * np.clip(y12 - y11, 0, None)
    area2 = np.clip(x22 - x21, 0, None) * np.clip(y22 - y21, 0, None)
    union = area1 + area2.T - inter + 1e-9
    return (inter / union).astype(np.float32)


def greedy_match(ious: np.ndarray, iou_thr: float, valid_pairs: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    triples: List[Tuple[float, int, int]] = []
    for gi in range(ious.shape[0]):
        for pj in range(ious.shape[1]):
            if ious[gi, pj] < iou_thr:
                continue
            if valid_pairs is not None and not bool(valid_pairs[gi, pj]):
                continue
            triples.append((float(ious[gi, pj]), gi, pj))
    triples.sort(reverse=True, key=lambda t: t[0])

    used_gt: set[int] = set()
    used_pr: set[int] = set()
    matches: List[Tuple[int, int]] = []
    for _, gi, pj in triples:
        if gi in used_gt or pj in used_pr:
            continue
        used_gt.add(gi)
        used_pr.add(pj)
        matches.append((gi, pj))
    return matches


def confusion_matrix_with_background(samples: Sequence[Dict[str, Any]], n_classes: int, score_threshold: float, iou_thr: float) -> np.ndarray:
    bg = n_classes
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
    for s in samples:
        gt_boxes = s["gt_boxes"]
        gt_cls = s["gt_cls"]

        keep = s["pred_scores"] >= score_threshold
        pred_boxes = s["pred_boxes"][keep]
        pred_cls = s["pred_cls"][keep]

        ious = iou_matrix(gt_boxes, pred_boxes)
        matches = greedy_match(ious, iou_thr=iou_thr)
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


def detection_counts(samples: Sequence[Dict[str, Any]], score_threshold: float, iou_thr: float) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    for s in samples:
        gt_boxes = s["gt_boxes"]
        gt_cls = s["gt_cls"]

        keep = s["pred_scores"] >= score_threshold
        pred_boxes = s["pred_boxes"][keep]
        pred_cls = s["pred_cls"][keep]

        ious = iou_matrix(gt_boxes, pred_boxes)
        valid = gt_cls[:, None] == pred_cls[None, :] if len(gt_boxes) and len(pred_boxes) else None
        matches = greedy_match(ious, iou_thr=iou_thr, valid_pairs=valid)

        m = len(matches)
        tp += m
        fn += max(0, len(gt_boxes) - m)
        fp += max(0, len(pred_boxes) - m)
    return tp, fp, fn


def build_binary_curve_samples_for_class(samples: Sequence[Dict[str, Any]], class_idx: int, iou_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    y_true: List[int] = []
    y_score: List[float] = []

    for s in samples:
        gt_mask = s["gt_cls"] == class_idx
        pr_mask = s["pred_cls"] == class_idx

        gt_boxes = s["gt_boxes"][gt_mask]
        pred_boxes = s["pred_boxes"][pr_mask]
        pred_scores = s["pred_scores"][pr_mask]

        ious = iou_matrix(gt_boxes, pred_boxes)
        matches = greedy_match(ious, iou_thr=iou_thr)
        matched_gt = {gi for gi, _ in matches}
        matched_pr = {pj for _, pj in matches}

        for gi, pj in matches:
            y_true.append(1)
            y_score.append(float(pred_scores[pj]))
        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                y_true.append(1)
                y_score.append(0.0)
        for pj in range(len(pred_boxes)):
            if pj not in matched_pr:
                y_true.append(0)
                y_score.append(float(pred_scores[pj]))

    return np.asarray(y_true, dtype=np.int64), np.asarray(y_score, dtype=np.float32)


def build_binary_curve_samples_overall(samples: Sequence[Dict[str, Any]], n_classes: int, iou_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    ys_true: List[np.ndarray] = []
    ys_score: List[np.ndarray] = []
    for class_idx in range(n_classes):
        y_true, y_score = build_binary_curve_samples_for_class(samples, class_idx, iou_thr)
        ys_true.append(y_true)
        ys_score.append(y_score)
    if not ys_true:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)
    return np.concatenate(ys_true), np.concatenate(ys_score)


def make_threshold_grid(all_scores: np.ndarray, n_points: int) -> np.ndarray:
    if all_scores.size == 0:
        return np.asarray([0.0, 1.0], dtype=np.float32)
    uniq = np.unique(np.concatenate([all_scores.astype(np.float32), np.asarray([0.0, 1.0], dtype=np.float32)]))
    if len(uniq) <= n_points:
        return uniq
    idx = np.linspace(0, len(uniq) - 1, n_points).round().astype(int)
    return uniq[idx]


def run_coco_eval(coco_gt: COCO, preds_path: Path, img_ids: Sequence[int], iou_thrs: np.ndarray) -> COCOeval:
    coco_dt = coco_gt.loadRes(str(preds_path)) if preds_path.exists() else coco_gt.loadRes([])
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.imgIds = list(img_ids)
    ev.params.iouThrs = iou_thrs.astype(np.float64)
    ev.evaluate()
    ev.accumulate()
    return ev


def ap_from_precision_slice(p: np.ndarray) -> float:
    valid = p[p > -1]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def ar_from_recall_slice(r: np.ndarray) -> float:
    valid = r[r > -1]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def find_iou_index(iou_thrs: np.ndarray, target: float, tol: float = 1e-6) -> Optional[int]:
    idx = np.where(np.abs(iou_thrs - target) <= tol)[0]
    return int(idx[0]) if idx.size else None


def save_confusion_csv(path: Path, labels: List[str], cm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + labels)
        for i, row in enumerate(cm):
            w.writerow([labels[i]] + row.tolist())


def draw_overlay(
    img_path: Path,
    gt_boxes: np.ndarray,
    gt_names: List[str],
    pred_boxes: np.ndarray,
    pred_names: List[str],
    pred_scores: np.ndarray,
    out_path: Path,
    image_max_side: int,
) -> None:
    img = Image.open(img_path).convert("RGB")
    scale = min(1.0, float(image_max_side) / float(max(img.size)))
    if scale < 0.999:
        new_wh = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_wh, Image.BILINEAR)
    draw = ImageDraw.Draw(img)

    def sc(box: np.ndarray) -> List[float]:
        return [float(box[0] * scale), float(box[1] * scale), float(box[2] * scale), float(box[3] * scale)]

    for b, n in zip(gt_boxes, gt_names):
        x1, y1, x2, y2 = sc(b)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1 + 2, y1 + 2), str(n), fill=(0, 255, 0))

    for b, n, s in zip(pred_boxes, pred_names, pred_scores):
        x1, y1, x2, y2 = sc(b)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1 + 2, y1 + 2), f"{float(s):.2f} {n}", fill=(255, 0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def try_plot_curves(output_dir: Path, thr_rows: Sequence[Dict[str, Any]], pr_rows: Sequence[Dict[str, Any]], roc_rows: Sequence[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib unavailable; skipping curve PNGs.")
        return

    if thr_rows:
        thr = [float(r["threshold"]) for r in thr_rows]
        p = [float(r["precision"]) for r in thr_rows]
        r = [float(r["recall"]) for r in thr_rows]
        f1 = [float(r["f1"]) for r in thr_rows]
        fppi = [float(r["fp_per_image"]) for r in thr_rows]

        plt.figure(figsize=(8, 5))
        plt.plot(thr, p, label="Precision")
        plt.plot(thr, r, label="Recall")
        plt.plot(thr, f1, label="F1")
        plt.xlabel("Score threshold")
        plt.ylabel("Metric")
        plt.title("Threshold Sweep")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "threshold_sweep.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(fppi, r, label="FROC")
        plt.xlabel("False positives per image")
        plt.ylabel("Sensitivity (Recall)")
        plt.title("FROC Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "froc_curve.png", dpi=180)
        plt.close()

    if pr_rows:
        rec = [float(r["recall"]) for r in pr_rows]
        prec = [float(r["precision"]) for r in pr_rows]
        plt.figure(figsize=(7, 5))
        plt.plot(rec, prec, label="PR")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "pr_curve_overall.png", dpi=180)
        plt.close()

    if roc_rows:
        fpr = [float(r["fpr"]) for r in roc_rows]
        tpr = [float(r["tpr"]) for r in roc_rows]
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], "--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (IoU-matched surrogate)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve_overall.png", dpi=180)
        plt.close()


def run_local_eval(cfg: LocalEvalConfig) -> None:
    ensure_deps()

    if not cfg.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {cfg.checkpoint}")
    if not cfg.test_json.exists():
        raise FileNotFoundError(f"test_json not found: {cfg.test_json}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    coco = COCO(str(cfg.test_json))
    cat_id_to_name, id_to_idx, idx_to_id = coco_categories(coco)
    n_classes = len(id_to_idx)
    labels = [cat_id_to_name[idx_to_id[i]] for i in range(n_classes)]

    model = load_model(cfg.model_class, cfg.checkpoint)

    img_ids = coco.getImgIds()
    if cfg.max_images is not None:
        img_ids = img_ids[: cfg.max_images]

    try:
        from tqdm import tqdm

        iterator = tqdm(img_ids, desc="Infer", unit="img")
    except Exception:
        iterator = img_ids

    preds_coco: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for img_id in iterator:
        info = coco.loadImgs([img_id])[0]
        file_name = str(info.get("file_name", ""))
        try:
            img_path = resolve_image_path_local(file_name, cfg.test_json, cfg.images_root)
        except FileNotFoundError as exc:
            if cfg.skip_missing_images:
                missing.append({"image_id": int(img_id), "file_name": file_name, "error": str(exc)})
                continue
            raise

        image = Image.open(img_path).convert("RGB")
        pred_boxes, pred_scores, pred_labels = predict_one_image(model, image, cfg.score_floor)
        pred_cat_ids = to_coco_cat_ids(pred_labels, cat_id_to_name, n_classes, idx_to_id)

        keep_known = np.array([int(cid) in id_to_idx for cid in pred_cat_ids], dtype=bool)
        pred_boxes = pred_boxes[keep_known]
        pred_scores = pred_scores[keep_known]
        pred_cat_ids = pred_cat_ids[keep_known]
        pred_cls = np.array([id_to_idx[int(cid)] for cid in pred_cat_ids], dtype=np.int64)

        for box, score, cid in zip(pred_boxes, pred_scores, pred_cat_ids):
            preds_coco.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(cid),
                    "score": float(score),
                    "bbox": to_coco_bbox_xywh(box.astype(float)),
                }
            )

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = [a for a in coco.loadAnns(ann_ids) if int(a.get("iscrowd", 0)) == 0]
        gt_boxes: List[List[float]] = []
        gt_cls: List[int] = []
        for a in anns:
            cid = int(a["category_id"])
            if cid not in id_to_idx:
                continue
            x, y, w, h = a["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
            gt_cls.append(id_to_idx[cid])

        samples.append(
            {
                "img_id": int(img_id),
                "img_path": str(img_path),
                "gt_boxes": np.asarray(gt_boxes, dtype=np.float32),
                "gt_cls": np.asarray(gt_cls, dtype=np.int64),
                "pred_boxes": pred_boxes.astype(np.float32),
                "pred_scores": pred_scores.astype(np.float32),
                "pred_cls": pred_cls.astype(np.int64),
            }
        )

    processed_ids = [int(s["img_id"]) for s in samples]
    if not processed_ids:
        raise RuntimeError("No images processed. Verify local test JSON/image paths.")

    preds_path = cfg.output_dir / "predictions_coco.json"
    json_dump(preds_path, preds_coco)

    iou_thrs = np.round(np.arange(cfg.iou_min, cfg.iou_max + cfg.iou_step * 0.5, cfg.iou_step), 2)
    std_iou = np.linspace(0.50, 0.95, 10, dtype=np.float64)

    coco_std = run_coco_eval(coco, preds_path, processed_ids, std_iou)
    coco_std.summarize()
    coco_sweep = run_coco_eval(coco, preds_path, processed_ids, iou_thrs)

    precision = coco_sweep.eval["precision"]
    recall = coco_sweep.eval["recall"]
    cat_ids_eval = list(coco_sweep.params.catIds)

    iou_rows: List[Dict[str, Any]] = []
    per_class_iou_rows: List[Dict[str, Any]] = []
    for t_idx, thr in enumerate(iou_thrs):
        iou_rows.append(
            {
                "iou_threshold": float(thr),
                "AP_all": ap_from_precision_slice(precision[t_idx, :, :, 0, -1]),
                "AR_all": ar_from_recall_slice(recall[t_idx, :, 0, -1]),
            }
        )
        for k, cid in enumerate(cat_ids_eval):
            per_class_iou_rows.append(
                {
                    "iou_threshold": float(thr),
                    "category_id": int(cid),
                    "class": cat_id_to_name[int(cid)],
                    "AP": ap_from_precision_slice(precision[t_idx, :, k, 0, -1]),
                    "AR": ar_from_recall_slice(recall[t_idx, k, 0, -1]),
                }
            )

    idx50 = find_iou_index(iou_thrs, 0.50)
    idx75 = find_iou_index(iou_thrs, 0.75)

    per_class_rows: List[Dict[str, Any]] = []
    for k, cid in enumerate(cat_ids_eval):
        per_class_rows.append(
            {
                "category_id": int(cid),
                "class": cat_id_to_name[int(cid)],
                "AP_iou_sweep": ap_from_precision_slice(precision[:, :, k, 0, -1]),
                "AR_iou_sweep": ar_from_recall_slice(recall[:, k, 0, -1]),
                "AP@50": ap_from_precision_slice(precision[idx50, :, k, 0, -1]) if idx50 is not None else float("nan"),
                "AR@50": ar_from_recall_slice(recall[idx50, k, 0, -1]) if idx50 is not None else float("nan"),
                "AP@75": ap_from_precision_slice(precision[idx75, :, k, 0, -1]) if idx75 is not None else float("nan"),
                "AR@75": ar_from_recall_slice(recall[idx75, k, 0, -1]) if idx75 is not None else float("nan"),
            }
        )

    write_csv(cfg.output_dir / "iou_sweep_metrics.csv", ["iou_threshold", "AP_all", "AR_all"], iou_rows)
    write_csv(cfg.output_dir / "per_class_iou_sweep.csv", ["iou_threshold", "category_id", "class", "AP", "AR"], per_class_iou_rows)
    write_csv(cfg.output_dir / "per_class_metrics.csv", ["category_id", "class", "AP_iou_sweep", "AR_iou_sweep", "AP@50", "AR@50", "AP@75", "AR@75"], per_class_rows)

    cm = confusion_matrix_with_background(samples, n_classes, cfg.score_threshold, cfg.confmat_iou)
    cm_labels = labels + ["background"]
    save_confusion_csv(cfg.output_dir / "confusion_matrix.csv", cm_labels, cm)
    json_dump(
        cfg.output_dir / "confusion_matrix.json",
        {
            "labels": cm_labels,
            "matrix": cm.tolist(),
            "score_threshold": cfg.score_threshold,
            "iou_threshold": cfg.confmat_iou,
        },
    )

    all_scores = np.concatenate([s["pred_scores"] for s in samples]) if samples else np.asarray([], dtype=np.float32)
    thresholds = make_threshold_grid(all_scores, cfg.threshold_points)
    thr_rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        tp, fp, fn = detection_counts(samples, float(thr), cfg.curve_iou)
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 0.0
        thr_rows.append(
            {
                "threshold": float(thr),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "fp_per_image": float(fp / max(1, len(samples))),
            }
        )
    write_csv(cfg.output_dir / "threshold_metrics.csv", ["threshold", "tp", "fp", "fn", "precision", "recall", "f1", "fp_per_image"], thr_rows)
    best_f1_row = max(thr_rows, key=lambda r: float(r["f1"])) if thr_rows else None

    pr_rows: List[Dict[str, Any]] = []
    roc_rows: List[Dict[str, Any]] = []
    pr_auc = float("nan")
    roc_auc = float("nan")

    y_true, y_score = build_binary_curve_samples_overall(samples, n_classes, cfg.curve_iou)
    if precision_recall_curve is not None and average_precision_score is not None and y_true.size > 0:
        p_vals, r_vals, p_thr = precision_recall_curve(y_true, y_score)
        pr_auc = float(average_precision_score(y_true, y_score))
        for i in range(len(p_vals)):
            thr = float(p_thr[i - 1]) if i > 0 and i - 1 < len(p_thr) else float("nan")
            pr_rows.append({"point_index": i, "threshold": thr, "precision": float(p_vals[i]), "recall": float(r_vals[i])})
        write_csv(cfg.output_dir / "pr_curve_overall.csv", ["point_index", "threshold", "precision", "recall"], pr_rows)

    if roc_curve is not None and auc is not None and y_true.size > 0 and len(np.unique(y_true)) > 1:
        fpr, tpr, thr = roc_curve(y_true, y_score)
        roc_auc = float(auc(fpr, tpr))
        for i in range(len(fpr)):
            roc_rows.append({"point_index": i, "threshold": float(thr[i]), "fpr": float(fpr[i]), "tpr": float(tpr[i])})
        write_csv(cfg.output_dir / "roc_curve_overall.csv", ["point_index", "threshold", "fpr", "tpr"], roc_rows)

    if not cfg.no_plots:
        try_plot_curves(cfg.output_dir, thr_rows, pr_rows, roc_rows)

    if cfg.num_overlays > 0:
        eligible = [s for s in samples if len(s["gt_boxes"]) > 0 or np.any(s["pred_scores"] >= cfg.score_threshold)]
        random.shuffle(eligible)
        chosen = eligible[: cfg.num_overlays]
        for i, s in enumerate(chosen, start=1):
            keep = s["pred_scores"] >= cfg.score_threshold
            pred_boxes = s["pred_boxes"][keep]
            pred_cls = s["pred_cls"][keep]
            pred_scores = s["pred_scores"][keep]
            gt_names = [labels[int(c)] for c in s["gt_cls"]]
            pred_names = [labels[int(c)] for c in pred_cls]
            draw_overlay(
                img_path=Path(s["img_path"]),
                gt_boxes=s["gt_boxes"],
                gt_names=gt_names,
                pred_boxes=pred_boxes,
                pred_names=pred_names,
                pred_scores=pred_scores,
                out_path=cfg.output_dir / "overlays" / f"overlay_{i:02d}.jpg",
                image_max_side=cfg.image_max_side,
            )

    coco_std_summary = {
        "AP@50:95": float(coco_std.stats[0]),
        "AP@50": float(coco_std.stats[1]),
        "AP@75": float(coco_std.stats[2]),
        "AR@1": float(coco_std.stats[6]),
        "AR@10": float(coco_std.stats[7]),
        "AR@100": float(coco_std.stats[8]),
    }

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": "local_only",
        "checkpoint": str(cfg.checkpoint),
        "test_json": str(cfg.test_json),
        "output_dir": str(cfg.output_dir),
        "images_root": str(cfg.images_root) if cfg.images_root is not None else None,
        "model_class": cfg.model_class,
        "images_total_requested": len(img_ids),
        "images_processed": len(processed_ids),
        "images_missing": len(missing),
        "missing_images": missing,
        "score_floor": cfg.score_floor,
        "score_threshold": cfg.score_threshold,
        "confmat_iou": cfg.confmat_iou,
        "curve_iou": cfg.curve_iou,
        "iou_sweep": {"min": cfg.iou_min, "max": cfg.iou_max, "step": cfg.iou_step, "rows": iou_rows},
        "coco_standard": coco_std_summary,
        "best_f1": best_f1_row,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "roc_note": "ROC/PR are computed from IoU-matched detection samples.",
    }
    json_dump(cfg.output_dir / "eval_summary.json", summary)

    print("\n[LOCAL EVAL] Done")
    print(f"[LOCAL EVAL] AP@50:95={coco_std_summary['AP@50:95']:.4f} AP@50={coco_std_summary['AP@50']:.4f} AP@75={coco_std_summary['AP@75']:.4f}")
    if best_f1_row is not None:
        print(f"[LOCAL EVAL] Best F1={float(best_f1_row['f1']):.4f} at threshold={float(best_f1_row['threshold']):.4f}")
    print(f"[LOCAL EVAL] Summary -> {(cfg.output_dir / 'eval_summary.json').resolve()}")


def build_config(args: argparse.Namespace) -> LocalEvalConfig:
    if args.checkpoint is None:
        raise ValueError("CHECKPOINT is not set. Edit CHECKPOINT at top or pass --checkpoint.")
    if args.test_json is None:
        raise ValueError("TEST_JSON is not set. Edit TEST_JSON at top or pass --test-json.")
    if args.output_dir is None:
        raise ValueError("OUTPUT_DIR is not set. Edit OUTPUT_DIR at top or pass --output-dir.")

    checkpoint = args.checkpoint.resolve()
    test_json = args.test_json.resolve()
    output_dir = args.output_dir.resolve()
    images_root = args.images_root.resolve() if args.images_root is not None else None
    model_meta_root = checkpoint.parent.parent if checkpoint.parent.name.lower() == "rfdetr_run" else checkpoint.parent

    model_class = infer_model_class(model_meta_root, checkpoint) if args.model_class == "auto" else args.model_class

    if args.iou_step <= 0:
        raise ValueError("--iou-step must be > 0")
    if args.iou_min <= 0 or args.iou_max > 1.0 or args.iou_min >= args.iou_max:
        raise ValueError("Need 0 < --iou-min < --iou-max <= 1.0")
    if args.threshold_points < 2:
        raise ValueError("--threshold-points must be >= 2")

    return LocalEvalConfig(
        checkpoint=checkpoint,
        test_json=test_json,
        output_dir=output_dir,
        images_root=images_root,
        model_class=model_class,
        score_floor=float(args.score_floor),
        score_threshold=float(args.score_threshold),
        confmat_iou=float(args.confmat_iou),
        curve_iou=float(args.curve_iou),
        iou_min=float(args.iou_min),
        iou_max=float(args.iou_max),
        iou_step=float(args.iou_step),
        threshold_points=int(args.threshold_points),
        max_images=args.max_images,
        num_overlays=int(args.num_overlays),
        image_max_side=int(args.image_max_side),
        seed=int(args.seed),
        skip_missing_images=bool(args.skip_missing_images),
        no_plots=bool(args.no_plots),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local-only RF-DETR evaluator")
    p.add_argument("--checkpoint", type=Path, default=_optional_path(CHECKPOINT), help="Path to model checkpoint file.")
    p.add_argument("--test-json", type=Path, default=_optional_path(TEST_JSON), help="Path to COCO test annotations JSON.")
    p.add_argument("--output-dir", type=Path, default=_optional_path(OUTPUT_DIR), help="Directory to write evaluation outputs.")
    p.add_argument("--images-root", type=Path, default=_optional_path(IMAGE_ROOT), help="Prefix root for relative image file_name entries.")
    p.add_argument("--model-class", type=str, default="auto", choices=["auto", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"])

    p.add_argument("--score-floor", type=float, default=0.001, help="Prediction floor retained for curves/AP.")
    p.add_argument("--score-threshold", type=float, default=0.30, help="Threshold used for confusion matrix/overlays.")
    p.add_argument("--confmat-iou", type=float, default=0.50)
    p.add_argument("--curve-iou", type=float, default=0.50)

    p.add_argument("--iou-min", type=float, default=0.10)
    p.add_argument("--iou-max", type=float, default=0.95)
    p.add_argument("--iou-step", type=float, default=0.05)
    p.add_argument("--threshold-points", type=int, default=51)

    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--num-overlays", type=int, default=8)
    p.add_argument("--image-max-side", type=int, default=1600)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--skip-missing-images", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = build_config(args)
    run_local_eval(cfg)


if __name__ == "__main__":
    main()
