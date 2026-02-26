#!/usr/bin/env python
"""
EvalRFDETR_SOLO_TestSummary.py

Two modes:

1) summarize (legacy)
   Aggregate existing run outputs (hpo_record.json + results.json).

2) evaluate
   Run full inference on a COCO test split from a trained RF-DETR run and export
   configurable object-detection metrics:
   - COCO AP/AR (standard AP@50:95, AP@50, AP@75)
   - IoU-swept AP/AR from 0.10 to 0.95 (step configurable)
   - Confusion matrix with background class
   - Threshold sweep (precision/recall/F1 + FROC points)
   - PR and ROC-style curves from IoU-matched detections
   - Optional overlay images
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import os.path as op
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Optional imports for evaluate mode only (kept optional so summarize mode can run
# even if these dependencies are missing).
try:
    import numpy as np
except Exception:  # pragma: no cover - dependency gate
    np = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover - dependency gate
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageDraw
except Exception:  # pragma: no cover - dependency gate
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception:  # pragma: no cover - dependency gate
    COCO = None  # type: ignore[assignment]
    COCOeval = None  # type: ignore[assignment]

try:
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
except Exception:  # pragma: no cover - dependency gate
    auc = None  # type: ignore[assignment]
    average_precision_score = None  # type: ignore[assignment]
    precision_recall_curve = None  # type: ignore[assignment]
    roc_curve = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Shared path helpers
# -----------------------------------------------------------------------------

def _detect_user_base() -> str | None:
    aau = glob.glob("/work/Member Files:*")
    if aau:
        return op.basename(aau[0])
    sdu = [d for d in glob.glob("/work/*#*") if op.isdir(d)]
    return op.basename(sdu[0]) if sdu else None


def env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    return Path(raw) if raw else default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


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


# -----------------------------------------------------------------------------
# summarize mode (legacy functionality)
# -----------------------------------------------------------------------------

def _pick_split_metrics(results_json: dict, split: str = "test") -> Tuple[Optional[float], Optional[float]]:
    cm = results_json.get("class_map", {})
    rows = cm.get(split, []) or cm.get(split.capitalize(), [])
    if not rows:
        return None, None

    picked = None
    for row in rows:
        if row.get("class") == "all":
            picked = row
            break
    if picked is None:
        picked = rows[0]

    ap50 = picked.get("map@50")
    ap5095 = picked.get("map@50:95")
    return (float(ap50) if ap50 is not None else None,
            float(ap5095) if ap5095 is not None else None)


def collect_summary_rows(eval_root: Path, target_filter: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    eval_root = eval_root.resolve()
    print(f"[SUMMARY] Scanning runs under: {eval_root}")
    if not eval_root.exists():
        raise FileNotFoundError(f"EVAL_RUNS_ROOT does not exist: {eval_root}")

    for sub in sorted(eval_root.iterdir()):
        if not sub.is_dir():
            continue

        hpo_path = sub / "hpo_record.json"
        res_path = sub / "results.json"
        if not hpo_path.exists() or not res_path.exists():
            continue

        hpo = json.loads(hpo_path.read_text(encoding="utf-8"))
        results = json.loads(res_path.read_text(encoding="utf-8"))

        target = hpo.get("target", "unknown")
        if target_filter and target != target_filter:
            continue

        encoder_ckpt = (
            hpo.get("ENCODER_CKPT")
            or hpo.get("encoder_ckpt")
            or hpo.get("pretrained_backbone")
            or hpo.get("encoder_name")
        )
        backbone_type = "SSL_backbone" if encoder_ckpt else "standard_backbone"

        valid_ap50, valid_ap5095 = _pick_split_metrics(results, split="valid")
        test_ap50, test_ap5095 = _pick_split_metrics(results, split="test")

        if valid_ap50 is None:
            valid_ap50 = hpo.get("val_AP50")
        if valid_ap5095 is None:
            valid_ap5095 = hpo.get("val_mAP5095")

        rows.append(
            {
                "run_dir": sub.name,
                "target": target,
                "backbone": backbone_type,
                "train_fraction": float(hpo.get("TRAIN_FRACTION", 1.0)),
                "val_AP50": valid_ap50,
                "val_mAP50_95": valid_ap5095,
                "test_AP50": test_ap50,
                "test_mAP50_95": test_ap5095,
                "encoder_ckpt": encoder_ckpt,
                "output_dir": str(sub),
            }
        )

    rows.sort(key=lambda row: (row["backbone"], row["train_fraction"]))
    return rows


def print_markdown_summary(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        print("[SUMMARY] No runs collected.")
        return

    headers = [
        "run_dir",
        "backbone",
        "train_frac",
        "val_AP50",
        "val_mAP50_95",
        "test_AP50",
        "test_mAP50_95",
    ]
    print("\n### RFDETR Test Summary\n")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        vals = [
            row["run_dir"],
            row["backbone"],
            f"{row['train_fraction']:.3f}",
            f"{row['val_AP50']:.3f}" if row["val_AP50"] is not None else "NA",
            f"{row['val_mAP50_95']:.3f}" if row["val_mAP50_95"] is not None else "NA",
            f"{row['test_AP50']:.3f}" if row["test_AP50"] is not None else "NA",
            f"{row['test_mAP50_95']:.3f}" if row["test_mAP50_95"] is not None else "NA",
        ]
        print("| " + " | ".join(vals) + " |")


def run_summary_mode(args: argparse.Namespace) -> None:
    rows = collect_summary_rows(args.eval_runs_root, target_filter=args.target_filter)
    write_csv(
        args.out_csv,
        fieldnames=[
            "run_dir",
            "target",
            "backbone",
            "train_fraction",
            "val_AP50",
            "val_mAP50_95",
            "test_AP50",
            "test_mAP50_95",
            "encoder_ckpt",
            "output_dir",
        ],
        rows=rows,
    )
    json_dump(
        args.out_json,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "eval_root": str(args.eval_runs_root),
            "n_runs": len(rows),
            "rows": rows,
        },
    )
    print_markdown_summary(rows)
    print(f"[SUMMARY] CSV -> {args.out_csv.resolve()}")
    print(f"[SUMMARY] JSON -> {args.out_json.resolve()}")


# -----------------------------------------------------------------------------
# evaluate mode
# -----------------------------------------------------------------------------

@dataclass
class EvalConfig:
    run_dir: Path
    checkpoint: Path
    test_json: Path
    output_dir: Path
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
    path_rewrites: List[Tuple[str, str]]
    images_root: Optional[Path]
    skip_missing_images: bool
    no_plots: bool


def ensure_eval_dependencies() -> None:
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
        raise ImportError(
            "evaluate mode requires missing dependencies: "
            + ", ".join(missing)
            + ". Install them and re-run."
        )


def parse_path_rewrites(raw: str) -> List[Tuple[str, str]]:
    """
    Format:
      "from1=to1;from2=to2"
    """
    raw = raw.strip()
    if not raw:
        # Backward-compatible default from previous local evaluation script.
        return [
            ("/work/MatiasMose#8097/", "D:/PHD/PhdData/"),
            ("\\work\\MatiasMose#8097\\", "D:/PHD/PhdData/"),
        ]

    pairs: List[Tuple[str, str]] = []
    for chunk in raw.split(";"):
        part = chunk.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(
                f"Invalid path rewrite '{part}'. Use FROM=TO;FROM=TO format."
            )
        src, dst = part.split("=", 1)
        src = src.strip()
        dst = dst.strip()
        if not src or not dst:
            raise ValueError(f"Invalid path rewrite '{part}' (empty source or target).")
        pairs.append((src, dst))
    return pairs


def _apply_rewrites(path_str: str, rewrites: Sequence[Tuple[str, str]]) -> List[Path]:
    cands: List[Path] = []
    if not path_str:
        return cands

    norm = path_str.replace("\\", "/")
    cands.append(Path(norm))
    cands.append(Path(path_str))
    for src, dst in rewrites:
        src_norm = src.replace("\\", "/")
        dst_norm = dst.replace("\\", "/")
        if norm.startswith(src_norm):
            rem = norm[len(src_norm):]
            cands.append(Path(dst_norm + rem))
    return cands


def resolve_image_path(
    file_name: str,
    test_json: Path,
    rewrites: Sequence[Tuple[str, str]],
    images_root: Optional[Path],
) -> Path:
    seen: set[str] = set()
    candidates: List[Path] = []
    candidates.extend(_apply_rewrites(file_name, rewrites))

    rel = file_name.replace("\\", "/")
    if rel:
        candidates.append(test_json.parent / rel)
    if images_root is not None and rel:
        candidates.append(images_root / rel)
        candidates.append(images_root / Path(rel).name)
        marker = "CellScanData/"
        if marker in rel:
            tail = rel.split(marker, 1)[1]
            candidates.append(images_root / tail)
            candidates.append(images_root / "CellScanData" / tail)

    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            if cand.exists():
                return cand.resolve()
        except OSError:
            continue

    raise FileNotFoundError(
        f"Could not resolve image path for '{file_name}'. "
        f"Checked rewrites={rewrites}, test_json.parent={test_json.parent}, "
        f"images_root={images_root}."
    )


def infer_model_class(run_dir: Path, checkpoint: Path) -> str:
    meta_model = run_dir / "rfdetr_run" / "run_meta" / "model_architecture.json"
    if meta_model.exists():
        try:
            js = json.loads(meta_model.read_text(encoding="utf-8"))
            name = str(js.get("model_name", "")).strip()
            if name in {"RFDETRSmall", "RFDETRMedium", "RFDETRLarge"}:
                return name
        except Exception:
            pass

    meta_train = run_dir / "rfdetr_run" / "run_meta" / "train_kwargs.json"
    if meta_train.exists():
        try:
            js = json.loads(meta_train.read_text(encoding="utf-8"))
            name = str(js.get("RFDETR_MODEL_CLS", "")).strip()
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
    from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall

    name_to_cls = {
        "RFDETRSmall": RFDETRSmall,
        "RFDETRMedium": RFDETRMedium,
        "RFDETRLarge": RFDETRLarge,
    }
    if model_class not in name_to_cls:
        raise ValueError(
            f"Unsupported model class '{model_class}'. "
            f"Expected one of {sorted(name_to_cls)}."
        )

    model = name_to_cls[model_class](pretrain_weights=str(checkpoint))
    if hasattr(model, "optimize_for_inference"):
        try:
            model.optimize_for_inference()
        except Exception:
            pass
    return model



def predict_one_image(model: Any, img_pil: Image.Image, score_floor: float):
    """
    Returns:
      boxes_xyxy: float32 [N,4]
      scores:     float32 [N]
      labels:     int64   [N]
    """
    arr = np.array(img_pil.convert("RGB"))  # type: ignore[arg-type]
    ten = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # type: ignore[union-attr]

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
            raise RuntimeError("Model has no predict/infer/forward_inference/forward method.")

    # Case A: supervision.Detections
    try:
        import supervision as sv  # optional

        if isinstance(out, sv.Detections):
            boxes = out.xyxy.astype(np.float32)
            if getattr(out, "confidence", None) is None:
                scores = np.ones((len(boxes),), dtype=np.float32)
            else:
                scores = out.confidence.astype(np.float32)
            if getattr(out, "class_id", None) is None:
                labels = np.zeros((len(boxes),), dtype=np.int64)
            else:
                labels = out.class_id.astype(np.int64)
            keep = scores >= score_floor
            return boxes[keep], scores[keep], labels[keep]
    except Exception:
        pass

    if isinstance(out, (list, tuple)) and len(out) == 1:
        out = out[0]
    if isinstance(out, (list, tuple)) and len(out) == 3:
        boxes, scores, labels = out
        out = {"boxes": boxes, "scores": scores, "labels": labels}

    if not isinstance(out, dict):
        raise RuntimeError(f"Prediction output not recognized. Got type={type(out)}.")

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
                detections = out["detections"]
                boxes = detections.get("boxes") or detections.get("bboxes")
                scores = detections.get("scores")
                labels = detections.get("labels") or detections.get("classes")
            else:
                boxes = out[kb]
                scores = out.get(ks)
                labels = out.get(kl)
            break

    if boxes is None:
        raise RuntimeError(f"Unrecognized prediction keys: {list(out.keys())}")

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


def to_coco_cat_ids(
    pred_labels: np.ndarray,
    cat_id_to_name: Dict[int, str],
    n_classes: int,
    idx_to_id: Dict[int, int],
) -> np.ndarray:
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


def greedy_match(
    ious: np.ndarray,
    iou_thr: float,
    valid_pairs: Optional[np.ndarray] = None,
) -> List[Tuple[int, int]]:
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


def confusion_matrix_with_background(
    samples: Sequence[Dict[str, Any]],
    n_classes: int,
    score_threshold: float,
    iou_thr: float,
) -> np.ndarray:
    bg = n_classes
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
    for sample in samples:
        gt_boxes = sample["gt_boxes"]
        gt_cls = sample["gt_cls"]

        keep = sample["pred_scores"] >= score_threshold
        pred_boxes = sample["pred_boxes"][keep]
        pred_cls = sample["pred_cls"][keep]

        ious = iou_matrix(gt_boxes, pred_boxes)
        matches = greedy_match(ious, iou_thr=iou_thr, valid_pairs=None)
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


def detection_counts(
    samples: Sequence[Dict[str, Any]],
    score_threshold: float,
    iou_thr: float,
) -> Tuple[int, int, int]:
    """
    Class-aware matching.
    Returns (tp, fp, fn).
    """
    tp = fp = fn = 0
    for sample in samples:
        gt_boxes = sample["gt_boxes"]
        gt_cls = sample["gt_cls"]

        keep = sample["pred_scores"] >= score_threshold
        pred_boxes = sample["pred_boxes"][keep]
        pred_cls = sample["pred_cls"][keep]

        ious = iou_matrix(gt_boxes, pred_boxes)
        if len(gt_boxes) and len(pred_boxes):
            valid = gt_cls[:, None] == pred_cls[None, :]
        else:
            valid = None
        matches = greedy_match(ious, iou_thr=iou_thr, valid_pairs=valid)

        m = len(matches)
        tp += m
        fn += max(0, len(gt_boxes) - m)
        fp += max(0, len(pred_boxes) - m)
    return tp, fp, fn


def build_binary_curve_samples_for_class(
    samples: Sequence[Dict[str, Any]],
    class_idx: int,
    iou_thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a class:
    - positives: GT instances (score=matched prediction score else 0)
    - negatives: unmatched predictions (score=prediction score)
    """
    y_true: List[int] = []
    y_score: List[float] = []

    for sample in samples:
        gt_mask = sample["gt_cls"] == class_idx
        pr_mask = sample["pred_cls"] == class_idx

        gt_boxes = sample["gt_boxes"][gt_mask]
        pred_boxes = sample["pred_boxes"][pr_mask]
        pred_scores = sample["pred_scores"][pr_mask]

        ious = iou_matrix(gt_boxes, pred_boxes)
        matches = greedy_match(ious, iou_thr=iou_thr, valid_pairs=None)
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


def build_binary_curve_samples_overall(
    samples: Sequence[Dict[str, Any]],
    n_classes: int,
    iou_thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
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

    for box, name in zip(gt_boxes, gt_names):
        x1, y1, x2, y2 = sc(box)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1 + 2, y1 + 2), str(name), fill=(0, 255, 0))

    for box, name, score in zip(pred_boxes, pred_names, pred_scores):
        x1, y1, x2, y2 = sc(box)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1 + 2, y1 + 2), f"{float(score):.2f} {name}", fill=(255, 0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_confusion_csv(path: Path, labels: List[str], cm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)
        for idx, row in enumerate(cm):
            writer.writerow([labels[idx]] + row.tolist())


def try_plot_curves(
    output_dir: Path,
    threshold_rows: Sequence[Dict[str, Any]],
    pr_rows: Sequence[Dict[str, Any]],
    roc_rows: Sequence[Dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[EVALUATE][WARN] matplotlib not available; skipping PNG curve plots.")
        return

    if threshold_rows:
        thr = [float(r["threshold"]) for r in threshold_rows]
        precision_vals = [float(r["precision"]) for r in threshold_rows]
        recall_vals = [float(r["recall"]) for r in threshold_rows]
        f1_vals = [float(r["f1"]) for r in threshold_rows]
        fp_per_img = [float(r["fp_per_image"]) for r in threshold_rows]

        plt.figure(figsize=(8, 5))
        plt.plot(thr, precision_vals, label="Precision")
        plt.plot(thr, recall_vals, label="Recall")
        plt.plot(thr, f1_vals, label="F1")
        plt.xlabel("Score threshold")
        plt.ylabel("Metric")
        plt.title("Threshold Sweep")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "threshold_sweep.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(fp_per_img, recall_vals, label="FROC")
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
        plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (IoU-matched surrogate)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve_overall.png", dpi=180)
        plt.close()


def run_coco_eval(
    coco_gt: COCO,
    results_path: Path,
    img_ids: Sequence[int],
    iou_thrs: np.ndarray,
) -> COCOeval:
    if results_path.exists():
        coco_dt = coco_gt.loadRes(str(results_path))
    else:
        coco_dt = coco_gt.loadRes([])
    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.imgIds = list(img_ids)
    evaluator.params.iouThrs = iou_thrs.astype(np.float64)
    evaluator.evaluate()
    evaluator.accumulate()
    return evaluator


def ap_from_precision_slice(precision_slice: np.ndarray) -> float:
    valid = precision_slice[precision_slice > -1]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def ar_from_recall_slice(recall_slice: np.ndarray) -> float:
    valid = recall_slice[recall_slice > -1]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def find_iou_index(iou_thrs: np.ndarray, target: float, tol: float = 1e-6) -> Optional[int]:
    idx = np.where(np.abs(iou_thrs - target) <= tol)[0]
    if idx.size == 0:
        return None
    return int(idx[0])


def build_eval_config(args: argparse.Namespace) -> EvalConfig:
    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint or (run_dir / "rfdetr_run" / "checkpoint_best_total.pth")
    checkpoint = checkpoint.resolve()
    test_json = args.test_json or (run_dir / "test" / "_annotations.coco.json")
    test_json = test_json.resolve()
    output_dir = args.output_dir or (run_dir / "rfdetr_run" / "eval_custom")
    output_dir = output_dir.resolve()

    if args.model_class == "auto":
        model_class = infer_model_class(run_dir, checkpoint)
    else:
        model_class = args.model_class

    if args.iou_step <= 0:
        raise ValueError("--iou-step must be > 0")
    if args.iou_min <= 0 or args.iou_max > 1.0 or args.iou_min >= args.iou_max:
        raise ValueError("--iou-min/--iou-max must satisfy 0 < iou_min < iou_max <= 1.0")
    if args.threshold_points < 2:
        raise ValueError("--threshold-points must be >= 2")

    rewrites = parse_path_rewrites(args.path_rewrite)

    return EvalConfig(
        run_dir=run_dir,
        checkpoint=checkpoint,
        test_json=test_json,
        output_dir=output_dir,
        model_class=model_class,
        score_floor=float(args.score_floor),
        score_threshold=float(args.score_threshold),
        confmat_iou=float(args.confmat_iou),
        curve_iou=float(args.curve_iou),
        iou_min=float(args.iou_min),
        iou_max=float(args.iou_max),
        iou_step=float(args.iou_step),
        threshold_points=int(args.threshold_points),
        max_images=int(args.max_images) if args.max_images is not None else None,
        num_overlays=int(args.num_overlays),
        image_max_side=int(args.image_max_side),
        seed=int(args.seed),
        path_rewrites=rewrites,
        images_root=args.images_root.resolve() if args.images_root is not None else None,
        skip_missing_images=bool(args.skip_missing_images),
        no_plots=bool(args.no_plots),
    )


def run_evaluate_mode(args: argparse.Namespace) -> None:
    ensure_eval_dependencies()
    cfg = build_eval_config(args)

    if not cfg.run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {cfg.run_dir}")
    if not cfg.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {cfg.checkpoint}")
    if not cfg.test_json.exists():
        raise FileNotFoundError(f"test_json not found: {cfg.test_json}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print("[EVALUATE] run_dir      :", cfg.run_dir)
    print("[EVALUATE] checkpoint   :", cfg.checkpoint)
    print("[EVALUATE] test_json    :", cfg.test_json)
    print("[EVALUATE] output_dir   :", cfg.output_dir)
    print("[EVALUATE] model_class  :", cfg.model_class)
    print("[EVALUATE] iou sweep    :", f"{cfg.iou_min:.2f}..{cfg.iou_max:.2f} step {cfg.iou_step:.2f}")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    coco = COCO(str(cfg.test_json))
    cat_id_to_name, id_to_idx, idx_to_id = coco_categories(coco)
    n_classes = len(id_to_idx)
    labels = [cat_id_to_name[idx_to_id[i]] for i in range(n_classes)]

    model = load_model(cfg.model_class, cfg.checkpoint)

    all_img_ids = coco.getImgIds()
    if cfg.max_images is not None:
        all_img_ids = all_img_ids[: cfg.max_images]

    try:
        from tqdm import tqdm
        iterator = tqdm(all_img_ids, desc="Infer", unit="img")
    except Exception:
        iterator = all_img_ids

    coco_results: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    missing_images: List[Dict[str, Any]] = []

    for img_id in iterator:
        info = coco.loadImgs([img_id])[0]
        file_name = str(info.get("file_name", ""))
        try:
            img_path = resolve_image_path(
                file_name=file_name,
                test_json=cfg.test_json,
                rewrites=cfg.path_rewrites,
                images_root=cfg.images_root,
            )
        except FileNotFoundError as exc:
            if cfg.skip_missing_images:
                missing_images.append({"image_id": int(img_id), "file_name": file_name, "error": str(exc)})
                continue
            raise

        image = Image.open(img_path).convert("RGB")
        pred_boxes, pred_scores, pred_labels = predict_one_image(model, image, cfg.score_floor)
        pred_cat_ids = to_coco_cat_ids(pred_labels, cat_id_to_name, n_classes, idx_to_id)

        # Keep only predictions mappable to known categories.
        keep_known = np.array([int(cid) in id_to_idx for cid in pred_cat_ids], dtype=bool)
        pred_boxes = pred_boxes[keep_known]
        pred_scores = pred_scores[keep_known]
        pred_cat_ids = pred_cat_ids[keep_known]
        pred_cls = np.array([id_to_idx[int(cid)] for cid in pred_cat_ids], dtype=np.int64)

        for box, score, cat_id in zip(pred_boxes, pred_scores, pred_cat_ids):
            coco_results.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(cat_id),
                    "score": float(score),
                    "bbox": to_coco_bbox_xywh(box.astype(float)),
                }
            )

        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = [a for a in coco.loadAnns(ann_ids) if int(a.get("iscrowd", 0)) == 0]
        gt_boxes: List[List[float]] = []
        gt_cls: List[int] = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cat_id = int(ann["category_id"])
            if cat_id not in id_to_idx:
                continue
            gt_boxes.append([x, y, x + w, y + h])
            gt_cls.append(id_to_idx[cat_id])

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

    processed_img_ids = [int(s["img_id"]) for s in samples]
    if not processed_img_ids:
        raise RuntimeError("No images were processed. Check test JSON paths / rewrite settings.")

    predictions_path = cfg.output_dir / "predictions_coco.json"
    json_dump(predictions_path, coco_results)

    iou_thrs = np.round(
        np.arange(cfg.iou_min, cfg.iou_max + cfg.iou_step * 0.5, cfg.iou_step),
        2,
    )
    std_iou_thrs = np.linspace(0.50, 0.95, 10, dtype=np.float64)

    coco_eval_std = run_coco_eval(coco, predictions_path, processed_img_ids, std_iou_thrs)
    coco_eval_std.summarize()
    coco_eval_sweep = run_coco_eval(coco, predictions_path, processed_img_ids, iou_thrs)

    precision = coco_eval_sweep.eval["precision"]  # [T, R, K, A, M]
    recall = coco_eval_sweep.eval["recall"]        # [T, K, A, M]
    cat_ids_eval = list(coco_eval_sweep.params.catIds)

    iou_rows: List[Dict[str, Any]] = []
    per_class_iou_rows: List[Dict[str, Any]] = []
    for t_idx, thr in enumerate(iou_thrs):
        ap_all = ap_from_precision_slice(precision[t_idx, :, :, 0, -1])
        ar_all = ar_from_recall_slice(recall[t_idx, :, 0, -1])
        iou_rows.append({"iou_threshold": float(thr), "AP_all": ap_all, "AR_all": ar_all})

        for k, cid in enumerate(cat_ids_eval):
            ap_cls = ap_from_precision_slice(precision[t_idx, :, k, 0, -1])
            ar_cls = ar_from_recall_slice(recall[t_idx, k, 0, -1])
            per_class_iou_rows.append(
                {
                    "iou_threshold": float(thr),
                    "category_id": int(cid),
                    "class": cat_id_to_name[int(cid)],
                    "AP": ap_cls,
                    "AR": ar_cls,
                }
            )

    idx50 = find_iou_index(iou_thrs, 0.50)
    idx75 = find_iou_index(iou_thrs, 0.75)

    per_class_rows: List[Dict[str, Any]] = []
    for k, cid in enumerate(cat_ids_eval):
        p_all = precision[:, :, k, 0, -1]
        r_all = recall[:, k, 0, -1]
        row: Dict[str, Any] = {
            "category_id": int(cid),
            "class": cat_id_to_name[int(cid)],
            "AP_iou_sweep": ap_from_precision_slice(p_all),
            "AR_iou_sweep": ar_from_recall_slice(r_all),
        }
        if idx50 is not None:
            row["AP@50"] = ap_from_precision_slice(precision[idx50, :, k, 0, -1])
            row["AR@50"] = ar_from_recall_slice(recall[idx50, k, 0, -1])
        else:
            row["AP@50"] = float("nan")
            row["AR@50"] = float("nan")
        if idx75 is not None:
            row["AP@75"] = ap_from_precision_slice(precision[idx75, :, k, 0, -1])
            row["AR@75"] = ar_from_recall_slice(recall[idx75, k, 0, -1])
        else:
            row["AP@75"] = float("nan")
            row["AR@75"] = float("nan")
        per_class_rows.append(row)

    write_csv(cfg.output_dir / "iou_sweep_metrics.csv", ["iou_threshold", "AP_all", "AR_all"], iou_rows)
    write_csv(
        cfg.output_dir / "per_class_iou_sweep.csv",
        ["iou_threshold", "category_id", "class", "AP", "AR"],
        per_class_iou_rows,
    )
    write_csv(
        cfg.output_dir / "per_class_metrics.csv",
        ["category_id", "class", "AP_iou_sweep", "AR_iou_sweep", "AP@50", "AR@50", "AP@75", "AR@75"],
        per_class_rows,
    )

    cm = confusion_matrix_with_background(
        samples=samples,
        n_classes=n_classes,
        score_threshold=cfg.score_threshold,
        iou_thr=cfg.confmat_iou,
    )
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
    threshold_rows: List[Dict[str, Any]] = []
    for thr in thresholds:
        tp, fp, fn = detection_counts(samples, score_threshold=float(thr), iou_thr=cfg.curve_iou)
        precision_thr = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall_thr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1_thr = float((2 * precision_thr * recall_thr) / (precision_thr + recall_thr)) if (precision_thr + recall_thr) > 0 else 0.0
        threshold_rows.append(
            {
                "threshold": float(thr),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": precision_thr,
                "recall": recall_thr,
                "f1": f1_thr,
                "fp_per_image": float(fp / max(1, len(samples))),
            }
        )
    write_csv(
        cfg.output_dir / "threshold_metrics.csv",
        ["threshold", "tp", "fp", "fn", "precision", "recall", "f1", "fp_per_image"],
        threshold_rows,
    )

    best_f1_row = max(threshold_rows, key=lambda r: float(r["f1"])) if threshold_rows else None

    pr_rows: List[Dict[str, Any]] = []
    roc_rows: List[Dict[str, Any]] = []
    pr_auc = float("nan")
    roc_auc = float("nan")
    roc_note = (
        "ROC/PR here are computed from IoU-matched detection samples "
        "(positives=GT objects, negatives=unmatched predictions)."
    )

    y_true, y_score = build_binary_curve_samples_overall(
        samples=samples,
        n_classes=n_classes,
        iou_thr=cfg.curve_iou,
    )
    if precision_recall_curve is not None and average_precision_score is not None and y_true.size > 0:
        prec, rec, pr_th = precision_recall_curve(y_true, y_score)
        pr_auc = float(average_precision_score(y_true, y_score))
        for i in range(len(prec)):
            thr = float(pr_th[i - 1]) if i > 0 and i - 1 < len(pr_th) else float("nan")
            pr_rows.append(
                {
                    "point_index": i,
                    "threshold": thr,
                    "precision": float(prec[i]),
                    "recall": float(rec[i]),
                }
            )
    else:
        print("[EVALUATE][WARN] sklearn precision-recall dependencies missing or no samples.")

    if roc_curve is not None and auc is not None and y_true.size > 0 and len(np.unique(y_true)) > 1:
        fpr, tpr, roc_th = roc_curve(y_true, y_score)
        roc_auc = float(auc(fpr, tpr))
        for i in range(len(fpr)):
            roc_rows.append(
                {
                    "point_index": i,
                    "threshold": float(roc_th[i]),
                    "fpr": float(fpr[i]),
                    "tpr": float(tpr[i]),
                }
            )
    else:
        print("[EVALUATE][WARN] ROC curve skipped (needs sklearn + both positive/negative classes).")

    if pr_rows:
        write_csv(cfg.output_dir / "pr_curve_overall.csv", ["point_index", "threshold", "precision", "recall"], pr_rows)
    if roc_rows:
        write_csv(cfg.output_dir / "roc_curve_overall.csv", ["point_index", "threshold", "fpr", "tpr"], roc_rows)

    if not cfg.no_plots:
        try_plot_curves(cfg.output_dir, threshold_rows, pr_rows, roc_rows)

    overlays_dir = cfg.output_dir / "overlays"
    if cfg.num_overlays > 0 and samples:
        eligible = [s for s in samples if len(s["gt_boxes"]) > 0 or np.any(s["pred_scores"] >= cfg.score_threshold)]
        random.shuffle(eligible)
        chosen = eligible[: cfg.num_overlays]
        for idx, sample in enumerate(chosen, start=1):
            keep = sample["pred_scores"] >= cfg.score_threshold
            pred_boxes = sample["pred_boxes"][keep]
            pred_cls = sample["pred_cls"][keep]
            pred_scores = sample["pred_scores"][keep]

            gt_names = [labels[int(c)] for c in sample["gt_cls"]]
            pred_names = [labels[int(c)] for c in pred_cls]
            draw_overlay(
                img_path=Path(sample["img_path"]),
                gt_boxes=sample["gt_boxes"],
                gt_names=gt_names,
                pred_boxes=pred_boxes,
                pred_names=pred_names,
                pred_scores=pred_scores,
                out_path=overlays_dir / f"overlay_{idx:02d}.jpg",
                image_max_side=cfg.image_max_side,
            )

    coco_std = {
        "AP@50:95": float(coco_eval_std.stats[0]),
        "AP@50": float(coco_eval_std.stats[1]),
        "AP@75": float(coco_eval_std.stats[2]),
        "AR@1": float(coco_eval_std.stats[6]),
        "AR@10": float(coco_eval_std.stats[7]),
        "AR@100": float(coco_eval_std.stats[8]),
    }

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(cfg.run_dir),
        "checkpoint": str(cfg.checkpoint),
        "test_json": str(cfg.test_json),
        "output_dir": str(cfg.output_dir),
        "model_class": cfg.model_class,
        "images_total_in_json": len(all_img_ids),
        "images_processed": len(processed_img_ids),
        "images_missing": len(missing_images),
        "missing_images": missing_images,
        "score_floor": cfg.score_floor,
        "score_threshold": cfg.score_threshold,
        "confmat_iou": cfg.confmat_iou,
        "curve_iou": cfg.curve_iou,
        "iou_sweep": {
            "min": cfg.iou_min,
            "max": cfg.iou_max,
            "step": cfg.iou_step,
            "rows": iou_rows,
        },
        "coco_standard": coco_std,
        "best_f1": best_f1_row,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "roc_note": roc_note,
        "outputs": {
            "predictions": str(predictions_path),
            "iou_sweep_metrics": str(cfg.output_dir / "iou_sweep_metrics.csv"),
            "per_class_metrics": str(cfg.output_dir / "per_class_metrics.csv"),
            "confusion_matrix_csv": str(cfg.output_dir / "confusion_matrix.csv"),
            "confusion_matrix_json": str(cfg.output_dir / "confusion_matrix.json"),
            "threshold_metrics": str(cfg.output_dir / "threshold_metrics.csv"),
            "pr_curve": str(cfg.output_dir / "pr_curve_overall.csv"),
            "roc_curve": str(cfg.output_dir / "roc_curve_overall.csv"),
        },
    }
    json_dump(cfg.output_dir / "eval_summary.json", summary)

    print("\n[EVALUATE] Done")
    print(f"[EVALUATE] AP@50:95={coco_std['AP@50:95']:.4f}  AP@50={coco_std['AP@50']:.4f}  AP@75={coco_std['AP@75']:.4f}")
    if best_f1_row is not None:
        print(
            "[EVALUATE] Best F1="
            f"{float(best_f1_row['f1']):.4f} at threshold={float(best_f1_row['threshold']):.4f} "
            f"(precision={float(best_f1_row['precision']):.4f}, recall={float(best_f1_row['recall']):.4f})"
        )
    print(f"[EVALUATE] Summary JSON -> {(cfg.output_dir / 'eval_summary.json').resolve()}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    user_base = os.environ.get("USER_BASE_DIR") or _detect_user_base() or ""
    work_root = Path("/work") / user_base if user_base else Path.cwd()

    parser = argparse.ArgumentParser(
        description="RF-DETR evaluation helper (legacy summary + full inference modes)."
    )
    sub = parser.add_subparsers(dest="mode")

    # summarize mode
    p_sum = sub.add_parser("summarize", help="Aggregate existing hpo_record/results files (legacy behavior).")
    default_eval_runs_root = env_path("EVAL_RUNS_ROOT", work_root / "RFDETR_FINAL_EVAL")
    p_sum.add_argument(
        "--eval-runs-root",
        type=Path,
        default=default_eval_runs_root,
        help="Root dir containing run subfolders with hpo_record.json + results.json.",
    )
    p_sum.add_argument(
        "--out-csv",
        type=Path,
        default=env_path("EVAL_OUT_CSV", default_eval_runs_root / "eval_summary.csv"),
    )
    p_sum.add_argument(
        "--out-json",
        type=Path,
        default=env_path("EVAL_OUT_JSON", default_eval_runs_root / "eval_summary.json"),
    )
    p_sum.add_argument(
        "--target-filter",
        type=str,
        default=os.getenv("EVAL_TARGET_FILTER", "").strip(),
        help="Optional exact class target filter.",
    )

    # evaluate mode
    p_eval = sub.add_parser("evaluate", help="Run inference on test set and compute detection metrics.")
    run_dir_env = os.getenv("EVAL_RUN_DIR", "").strip()
    ckpt_env = os.getenv("EVAL_CHECKPOINT", "").strip()
    test_json_env = os.getenv("EVAL_TEST_JSON", "").strip()
    output_env = os.getenv("EVAL_OUTPUT_DIR", "").strip()
    images_root_env = os.getenv("EVAL_IMAGES_ROOT", "").strip()

    p_eval.add_argument("--run-dir", type=Path, default=Path(run_dir_env) if run_dir_env else Path("."))
    p_eval.add_argument("--checkpoint", type=Path, default=Path(ckpt_env) if ckpt_env else None)
    p_eval.add_argument("--test-json", type=Path, default=Path(test_json_env) if test_json_env else None)
    p_eval.add_argument("--output-dir", type=Path, default=Path(output_env) if output_env else None)
    p_eval.add_argument(
        "--model-class",
        type=str,
        default=os.getenv("EVAL_MODEL_CLASS", "auto"),
        choices=["auto", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge"],
    )
    p_eval.add_argument("--score-floor", type=float, default=float(os.getenv("EVAL_SCORE_FLOOR", "0.001")))
    p_eval.add_argument("--score-threshold", type=float, default=float(os.getenv("EVAL_SCORE_THRESHOLD", "0.30")))
    p_eval.add_argument("--confmat-iou", type=float, default=float(os.getenv("EVAL_CONFMAT_IOU", "0.50")))
    p_eval.add_argument("--curve-iou", type=float, default=float(os.getenv("EVAL_CURVE_IOU", "0.50")))
    p_eval.add_argument("--iou-min", type=float, default=float(os.getenv("EVAL_IOU_MIN", "0.10")))
    p_eval.add_argument("--iou-max", type=float, default=float(os.getenv("EVAL_IOU_MAX", "0.95")))
    p_eval.add_argument("--iou-step", type=float, default=float(os.getenv("EVAL_IOU_STEP", "0.05")))
    p_eval.add_argument("--threshold-points", type=int, default=int(os.getenv("EVAL_THRESHOLD_POINTS", "51")))
    p_eval.add_argument("--max-images", type=int, default=None)
    p_eval.add_argument("--num-overlays", type=int, default=int(os.getenv("EVAL_NUM_OVERLAYS", "8")))
    p_eval.add_argument("--image-max-side", type=int, default=int(os.getenv("EVAL_IMAGE_MAX_SIDE", "1600")))
    p_eval.add_argument("--seed", type=int, default=int(os.getenv("EVAL_SEED", "42")))
    p_eval.add_argument(
        "--path-rewrite",
        type=str,
        default=os.getenv("EVAL_PATH_REWRITE", ""),
        help="Path rewrite rules in FROM=TO;FROM=TO format.",
    )
    p_eval.add_argument("--images-root", type=Path, default=Path(images_root_env) if images_root_env else None)
    p_eval.add_argument(
        "--skip-missing-images",
        action="store_true",
        default=env_bool("EVAL_SKIP_MISSING_IMAGES", False),
    )
    p_eval.add_argument("--no-plots", action="store_true", default=env_bool("EVAL_NO_PLOTS", False))

    return parser


def normalize_argv(argv: Sequence[str]) -> List[str]:
    """
    Backward compatibility:
    - no args -> summarize
    - args without explicit subcommand:
      - if they include evaluate-only flags -> evaluate
      - else summarize
    """
    if not argv:
        return ["summarize"]
    if len(argv) == 1 and argv[0] in {"-h", "--help"}:
        return list(argv)
    if argv[0] in {"summarize", "evaluate"}:
        return list(argv)
    eval_flags = {
        "--run-dir",
        "--checkpoint",
        "--test-json",
        "--output-dir",
        "--model-class",
        "--score-threshold",
        "--score-floor",
    }
    if any(flag in argv for flag in eval_flags):
        return ["evaluate"] + list(argv)
    return ["summarize"] + list(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(normalize_argv(sys.argv[1:] if argv is None else argv))

    if args.mode == "summarize":
        run_summary_mode(args)
        return
    if args.mode == "evaluate":
        run_evaluate_mode(args)
        return
    raise RuntimeError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
