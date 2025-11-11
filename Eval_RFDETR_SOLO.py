#!/usr/bin/env python3
"""
Evaluate an RF-DETR checkpoint on a COCO-format TEST set.

Outputs:
- eval_summary.json            # AP/AR (overall + per-class), confusion matrix counts
- coco_metrics.txt             # pretty COCOeval table
- overlays/*.jpg               # 4 example images with GT (green) and Pred (red)
- per_image_results.json       # raw predictions in COCO format
- per_class_metrics.csv        # per-class AP@0.50, AP@0.50:0.95, AR
"""

import json, math, os, random, csv
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# --- pip install: pycocotools matplotlib numpy pillow torch tqdm ---
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge

# ====== USER INPUTS ======
RunName = "dataset_coco_splits_20251030-103504_Base_AllClasses_Leucocyte"
CHECKPOINT = "./RFDETR_SOLO_OUTPUT/" + RunName + "/rfdetr_run/checkpoint_best_total.pth"
TEST_JSON  = Path(r"./RFDETR_SOLO_OUTPUT/" + RunName + "/test/_annotations.coco.json")
OUT_DIR    = Path("SOLO_Supervised_RFDETR/rfdeval_out") / RunName
SCORE_THRESH = 0.30     # confidence threshold for predictions & confusion/overlays
IOU_FOR_CONFMAT = 0.50  # IoU threshold used to match GT<->Pred for confusion matrix
MAX_IMAGES = None       # set e.g. 200 to evaluate on subset for speed
NUM_OVERLAYS = 10
IMAGE_MAX_SIDE = 1600   # overlay resize cap to avoid huge images
SEED = 42
# =========================

# ---------- RF-DETR inference shim ----------

@torch.no_grad()

# ───────────────────────────────────────────────────────────────
# Helper: normalize file paths (UCloud → local Windows)
def fix_ucloud_path(path_str: str) -> str:
    """
    If the COCO JSON image paths contain UCloud prefixes like:
        /work/MatiasMose#8097/CellScanData/...
    then automatically replace with your local Windows base path.
    """
    if not path_str:
        return path_str

    ucloud_prefix = "/work/MatiasMose#8097/"
    local_prefix  = r"D:/PHD/PhdData/"

    # Replace forward slashes and prefix if necessary
    if path_str.startswith(ucloud_prefix):
        path_str = path_str.replace(ucloud_prefix, local_prefix)
    elif path_str.startswith("\\work\\MatiasMose#8097\\"):
        path_str = path_str.replace("\\work\\MatiasMose#8097\\", local_prefix)

    # Always normalize to Windows-style separators
    path_str = os.path.normpath(path_str)
    return path_str
# ───────────────────────────────────────────────────────────────


def predict_one_image(model, img_pil: Image.Image, score_thresh: float):
    """
    Returns: boxes_xyxy [N,4], scores [N], class_idx [N]
    Supports RF-DETR outputs as either:
      - supervision.Detections (preferred by your build)
      - dict with keys like boxes/scores/labels or bboxes/scores/classes
    """
    import numpy as np
    import torch

    # --- prepare flexible inputs ---
    arr = np.array(img_pil.convert("RGB"))
    ten = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    # --- call the right method ---
    out = None
    for name in ("predict", "infer", "inference", "forward_inference"):
        fn = getattr(model, name, None)
        if callable(fn):
            # try PIL -> ndarray -> tensor
            for inp in (img_pil, arr, ten):
                try:
                    out = fn(inp)
                    break
                except Exception:
                    out = None
            if out is not None:
                break
    if out is None:
        # last resort: forward
        if hasattr(model, "forward") and callable(getattr(model, "forward")):
            out = model.forward(ten)
        else:
            raise RuntimeError(
                "Model has no predict/infer/forward_inference/forward method."
            )

    # --- normalize output ---
    # Case A: supervision.Detections
    try:
        import supervision as sv  # pip install supervision
        if isinstance(out, sv.Detections):
            boxes  = out.xyxy.astype(np.float32)
            # some versions may store None if confidence not provided
            scores = (
                out.confidence.astype(np.float32)
                if getattr(out, "confidence", None) is not None
                else np.ones((len(boxes),), dtype=np.float32)
            )
            labels = (
                out.class_id.astype(np.int64)
                if getattr(out, "class_id", None) is not None
                else np.zeros((len(boxes),), dtype=np.int64)
            )
            keep = scores >= score_thresh
            return boxes[keep], scores[keep], labels[keep]
    except Exception:
        # supervision not installed or not used — fall through to dict handling
        pass

    # Case B: dict-like outputs
    if isinstance(out, (list, tuple)) and len(out) == 1:
        out = out[0]
    if isinstance(out, (list, tuple)) and len(out) == 3:
        b, s, l = out
        out = {"boxes": b, "scores": s, "labels": l}

    if not isinstance(out, dict):
        raise RuntimeError(
            f"Prediction output not recognized. Got: {type(out)}. "
            "Install `supervision` (pip install supervision) or adapt this function."
        )

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
                boxes  = d.get("boxes") or d.get("bboxes")
                scores = d.get("scores")
                labels = d.get("labels") or d.get("classes")
            else:
                boxes, scores, labels = out[kb], out.get(ks), out.get(kl)
            break
    if boxes is None:
        raise RuntimeError(f"Unrecognized dict keys: {list(out.keys())}")

    to_np = lambda t: t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    boxes  = to_np(boxes).astype(np.float32)
    scores = to_np(scores).astype(np.float32) if scores is not None else np.ones((len(boxes),), dtype=np.float32)
    labels = to_np(labels).astype(np.int64)   if labels is not None else np.zeros((len(boxes),), dtype=np.int64)

    keep = scores >= score_thresh
    return boxes[keep], scores[keep], labels[keep]


# ---------- COCO helpers ----------
def coco_categories(coco: COCO):
    cats = coco.loadCats(coco.getCatIds())
    cat_id_to_name = {c["id"]: c["name"] for c in cats}
    cat_name_to_id = {c["name"]: c["id"] for c in cats}
    # RF-DETR label indices usually 0..C-1 in training order; we need a mapping.
    # We'll derive the order from categories in TEST JSON (typical RF-DETR expects same order).
    ordered_ids = [c["id"] for c in sorted(cats, key=lambda x: x["id"])]
    id_to_idx = {cid: i for i, cid in enumerate(ordered_ids)}
    idx_to_id = {i: cid for cid, i in id_to_idx.items()}
    return cat_id_to_name, id_to_idx, idx_to_id

def to_coco_bbox_xywh(box_xyxy: np.ndarray) -> List[float]:
    x1,y1,x2,y2 = box_xyxy.tolist()
    return [float(x1), float(y1), float(max(0.0,x2-x1)), float(max(0.0,y2-y1))]

# ---------- Confusion matrix (with background) ----------
def iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    # boxes: (N,4)/(M,4) xyxy
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    x11,y11,x12,y12 = np.split(boxes1, 4, axis=1)
    x21,y21,x22,y22 = np.split(boxes2, 4, axis=1)
    inter_x1 = np.maximum(x11, x21.T)
    inter_y1 = np.maximum(y11, y21.T)
    inter_x2 = np.minimum(x12, x22.T)
    inter_y2 = np.minimum(y12, y22.T)
    inter_w  = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h  = np.clip(inter_y2 - inter_y1, 0, None)
    inter    = inter_w * inter_h
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union = area1 + area2.T - inter + 1e-9
    return (inter / union).astype(np.float32)

def build_confusion(gt_boxes, gt_cls_ids, pr_boxes, pr_cls_ids, iou_thr, n_classes):
    """
    Returns a square (n_classes+1) matrix where the last index is 'background'.
    gt row vs pred col:
      - matched pairs go to (gt_class, pred_class)
      - unmatched GT -> (gt_class, background)  [FN]
      - unmatched Pred -> (background, pred_class)  [FP]
    """
    cm = np.zeros((n_classes+1, n_classes+1), dtype=np.int64)
    used_pred = set()
    ious = iou_matrix(gt_boxes, pr_boxes)

    # greedy matching per GT, highest IoU
    for gi in range(len(gt_boxes)):
        if pr_boxes.size == 0:
            cm[gt_cls_ids[gi], n_classes] += 1
            continue
        pj = int(np.argmax(ious[gi])) if ious.shape[1] else -1
        if pj >= 0 and ious[gi, pj] >= iou_thr and pj not in used_pred:
            used_pred.add(pj)
            cm[gt_cls_ids[gi], pr_cls_ids[pj]] += 1
        else:
            cm[gt_cls_ids[gi], n_classes] += 1  # FN

    # remaining predictions are FP
    for pj in range(len(pr_boxes)):
        if pj not in used_pred:
            cm[n_classes, pr_cls_ids[pj]] += 1
    return cm

# ---------- Overlay drawing ----------
def draw_overlays(img_path: Path, gt, pred, out_path: Path):
    img = Image.open(img_path).convert("RGB")
    # resize if too big
    scale = min(1.0, IMAGE_MAX_SIDE / max(img.size))
    if scale < 0.999:
        new_wh = (int(img.width*scale), int(img.height*scale))
        img = img.resize(new_wh, Image.BILINEAR)

    draw = ImageDraw.Draw(img)
    def scale_box(b):
        return [b[0]*scale, b[1]*scale, b[2]*scale, b[3]*scale]

    # GT: green
    for b, name in gt:
        x1,y1,x2,y2 = scale_box(b)
        draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=2)
        draw.text((x1+2, y1+2), name, fill=(0,255,0))
    # Pred: red
    for b, name, score in pred:
        x1,y1,x2,y2 = scale_box(b)
        draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=2)
        draw.text((x1+2, y1+2), f"{name} {score:.2f}", fill=(255,0,0))

    img.save(out_path)

def match_by_iou(gt_boxes, pr_boxes, iou_thr=0.5):
    """
    Greedy one-to-one matching by IoU (desc).
    Returns list of (gi, pj) pairs.
    """
    import numpy as np
    if len(gt_boxes) == 0 or len(pr_boxes) == 0:
        return []
    ious = iou_matrix(np.asarray(gt_boxes, dtype=np.float32),
                      np.asarray(pr_boxes, dtype=np.float32))
    # list of (iou, gi, pj) sorted high->low
    triples = []
    for gi in range(ious.shape[0]):
        for pj in range(ious.shape[1]):
            if ious[gi, pj] >= iou_thr:
                triples.append((ious[gi, pj], gi, pj))
    triples.sort(reverse=True, key=lambda t: t[0])

    matched_gt = set()
    matched_pr = set()
    matches = []
    for _, gi, pj in triples:
        if gi not in matched_gt and pj not in matched_pr:
            matched_gt.add(gi); matched_pr.add(pj)
            matches.append((gi, pj))
    return matches

def to_coco_cat_ids(pred_labels: np.ndarray, cat_id_to_name: dict, n_classes: int, idx_to_id: dict):
    """
    If labels already look like COCO category_ids (all present in cat_id_to_name),
    return them as-is. Otherwise treat them as 0..C-1 class indices and map via idx_to_id.
    """
    pred_labels = pred_labels.astype(int)
    keys_catids = set(cat_id_to_name.keys())
    if len(pred_labels) > 0 and np.all(np.isin(pred_labels, list(keys_catids))):
        # Already category_ids
        return pred_labels
    # Otherwise assume 0..C-1 indices
    if np.all((pred_labels >= 0) & (pred_labels < n_classes)):
        return np.array([idx_to_id[int(i)] for i in pred_labels], dtype=np.int64)
    # Fallback: clip to index range then map
    pred_labels = np.clip(pred_labels, 0, n_classes - 1)
    return np.array([idx_to_id[int(i)] for i in pred_labels], dtype=np.int64)



# ---------- Main ----------
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "overlays").mkdir(exist_ok=True)

    # Load COCO test set
    coco = COCO(str(TEST_JSON))
    cat_id_to_name, id_to_idx, idx_to_id = coco_categories(coco)
    n_classes = len(id_to_idx)

    # Load model

    model = RFDETRLarge(pretrain_weights=CHECKPOINT)
    #if hasattr(model, "optimize_for_inference"):
        #model.optimize_for_inference()

    # Iterate images & collect predictions in COCO format
    img_ids = coco.getImgIds()
    if MAX_IMAGES:
        img_ids = img_ids[:MAX_IMAGES]
    results = []
    cm_total = np.zeros((n_classes+1, n_classes+1), dtype=np.int64)

    overlay_samples = []
    for i, img_id in enumerate(tqdm(img_ids, desc="Infer")):
        im_meta = coco.loadImgs(img_id)[0]
        img_path = Path(fix_ucloud_path(im_meta["file_name"]))
        if not img_path.exists():
            # If file_name wasn’t absolute, interpret relative to JSON parent
            img_path = TEST_JSON.parent / im_meta["file_name"]
        im = Image.open(img_path).convert("RGB")

        # Predict
        boxes_xyxy, scores, cls_idx = predict_one_image(model, im, SCORE_THRESH)
        # Ensure predictions are COCO category_ids (robust)
        cat_ids = to_coco_cat_ids(cls_idx, cat_id_to_name, n_classes, idx_to_id)

        # Collect COCO-style results
        for b, s, cid in zip(boxes_xyxy, scores, cat_ids):
            results.append({
                "image_id": int(img_id),
                "category_id": int(cid),
                "score": float(s),
                "bbox": to_coco_bbox_xywh(b.astype(float)),
            })

        # Confusion matrix build (IoU_FOR_CONFMAT), use GT & predictions above threshold
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        gt_boxes = []
        gt_cls = []
        for a in anns:
            # skip crowd/ignore if present
            if a.get("iscrowd", 0) == 1:
                continue
            x,y,w,h = a["bbox"]
            gt_boxes.append([x, y, x+w, y+h])
            gt_cls.append(id_to_idx[a["category_id"]])
        gt_boxes = np.array(gt_boxes, dtype=np.float32)
        gt_cls = np.array(gt_cls, dtype=np.int64)

        pr_boxes = boxes_xyxy.astype(np.float32)
        pr_cls = np.array([int(id_to_idx[c]) for c in cat_ids], dtype=np.int64)

        cm = build_confusion(gt_boxes, gt_cls, pr_boxes, pr_cls, IOU_FOR_CONFMAT, n_classes)
        cm_total += cm

        # pick up to NUM_OVERLAYS diverse images
        if len(overlay_samples) < NUM_OVERLAYS and (len(gt_boxes) > 0 or len(pr_boxes) > 0):
            overlay_samples.append((img_path, gt_boxes.copy(), gt_cls.copy(), pr_boxes.copy(), pr_cls.copy(), scores.copy()))

    # Save per-image results
    (OUT_DIR / "per_image_results.json").write_text(json.dumps(results), encoding="utf-8")

    # COCOeval
    coco_dt = coco.loadRes(str(OUT_DIR / "per_image_results.json")) if results else coco.loadRes([])
    coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()

    # Persist the COCOeval table
    with (OUT_DIR / "coco_metrics.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join([f"{m:.4f}" if isinstance(m, float) else str(m) for m in coco_eval.stats]))

    # Per-class AP@0.50 and AP@[.50:.95]
    per_class = []
    for cid, cname in cat_id_to_name.items():
        coco_eval.params.catIds = [cid]
        coco_eval.evaluate(); coco_eval.accumulate()
        ap_50_95 = float(np.mean(coco_eval.eval["precision"][..., 0:10])) if coco_eval.eval and coco_eval.eval["precision"].size else float("nan")
        # For AP@0.50 specifically:
        coco_eval.params.iouThrs = np.array([0.50], dtype=np.float64)
        coco_eval.evaluate(); coco_eval.accumulate()
        ap_50 = float(np.mean(coco_eval.eval["precision"])) if coco_eval.eval and coco_eval.eval["precision"].size else float("nan")
        # Reset IOU thresholds
        coco_eval.params.iouThrs = np.linspace(.5, .95, 10)
        per_class.append({"category_id": cid, "class": cname, "AP@50:95": ap_50_95, "AP@50": ap_50})

    # Overall summary (COCOeval.stats order for bbox)
    # [0]=AP, [1]=AP50, [2]=AP75, [3]=AP_small, [4]=AP_medium, [5]=AP_large,
    # [6]=AR, [7]=AR_small, [8]=AR_medium, [9]=AR_large
    summary = {
        "AP@50:95_all": float(coco_eval.stats[0]),
        "AP@50_all":    float(coco_eval.stats[1]),
        "AP@75_all":    float(coco_eval.stats[2]),
        "AR@50:95_all": float(coco_eval.stats[6]),
        "per_class": per_class,
    }

    # Save per-class CSV
    with (OUT_DIR / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["category_id","class","AP@50:95","AP@50"])
        for r in per_class:
            w.writerow([r["category_id"], r["class"], f"{r['AP@50:95']:.4f}", f"{r['AP@50']:.4f}"])

    # Save confusion matrix
    labels = [cat_id_to_name[idx_to_id[i]] for i in range(n_classes)]
    cm_dict = {
        "labels": labels + ["background"],
        "matrix": cm_total.tolist(),
        "iou_for_confmat": IOU_FOR_CONFMAT,
        "score_thresh": SCORE_THRESH
    }
    (OUT_DIR / "confusion_matrix.json").write_text(json.dumps(cm_dict, indent=2), encoding="utf-8")

    # Overlays (pick GT & Pred labels)
    for k, (img_path, gtb, gtc, prb, prc, prs) in enumerate(overlay_samples):
        gt_pack = [(gtb[i], labels[gtc[i]]) for i in range(len(gtb))]
        pr_pack = [(prb[i], labels[prc[i]], float(prs[i])) for i in range(len(prb))]
        draw_overlays(img_path, gt_pack, pr_pack, OUT_DIR / "overlays" / f"overlay_{k+1}.jpg")

    # Final summary JSON
    out_json = {
        "coco_summary": summary,
        "confusion_matrix": cm_dict,
        "num_images_eval": len(img_ids),
        "checkpoint": str(CHECKPOINT),
        "test_json": str(TEST_JSON),
    }
    (OUT_DIR / "eval_summary.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    print("\n== Done ==")
    print(f"Saved: {OUT_DIR/'eval_summary.json'}")
    print(f"      {OUT_DIR/'confusion_matrix.json'}")
    print(f"      {OUT_DIR/'per_class_metrics.csv'}")
    print(f"      {OUT_DIR/'overlays'}")
    print(f"      {OUT_DIR/'coco_metrics.txt'}")

    print("\n[SCIKIT] Computing IoU-matched classification metrics with background…")

    BACKGROUND = n_classes  # last index
    y_true_all, y_pred_all = [], []

    for img_id in img_ids:
        # --- GT ---
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = [a for a in coco.loadAnns(ann_ids) if a.get("iscrowd", 0) == 0]
        gt_boxes = [[a["bbox"][0], a["bbox"][1],
                     a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
        gt_ids = [a["category_id"] for a in anns]  # COCO category_ids
        gt_idx = [id_to_idx[cid] for cid in gt_ids]  # 0..C-1

        # --- Predictions ---
        im_meta = coco.loadImgs(img_id)[0]
        img_path = Path(fix_ucloud_path(im_meta["file_name"]))
        if not img_path.exists():
            img_path = TEST_JSON.parent / im_meta["file_name"]
        im = Image.open(img_path).convert("RGB")
        pr_boxes_xyxy, pr_scores, pr_labels = predict_one_image(model, im, SCORE_THRESH)

        # Normalize labels → COCO category_ids, then → indices
        pr_cat_ids = to_coco_cat_ids(pr_labels, cat_id_to_name, n_classes, idx_to_id)
        pr_idx = [id_to_idx[int(cid)] for cid in pr_cat_ids]

        # --- IoU matching ---
        matches = match_by_iou(gt_boxes, pr_boxes_xyxy, iou_thr=IOU_FOR_CONFMAT)
        matched_gt = set(gi for gi, _ in matches)
        matched_pr = set(pj for _, pj in matches)

        # Matched pairs (count as normal class→class)
        for gi, pj in matches:
            y_true_all.append(gt_idx[gi])
            y_pred_all.append(pr_idx[pj])

        # Unmatched GT → (class, background)  (false negatives)
        for gi in range(len(gt_idx)):
            if gi not in matched_gt:
                y_true_all.append(gt_idx[gi])
                y_pred_all.append(BACKGROUND)

        # Unmatched Pred → (background, class)  (false positives)
        for pj in range(len(pr_idx)):
            if pj not in matched_pr:
                y_true_all.append(BACKGROUND)
                y_pred_all.append(pr_idx[pj])

    if len(y_true_all) == 0:
        print("[SCIKIT] No samples; skipping sklearn metrics.")
    else:
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        labels_sklearn = list(range(n_classes + 1))  # include background
        target_names = [cat_id_to_name[idx_to_id[i]] for i in range(n_classes)] + ["background"]

        # Report (background row will be included)
        report = classification_report(
            y_true_all, y_pred_all,
            labels=labels_sklearn, target_names=target_names,
            digits=3, zero_division=0, output_dict=True,
        )
        report_df = pd.DataFrame(report).transpose()
        report_csv_path = OUT_DIR / "classification_report_iou_bg.csv"
        report_df.to_csv(report_csv_path)
        print(f"[SCIKIT] Classification report saved to: {report_csv_path}")

        # Confusion matrix
        cm_sklearn = confusion_matrix(y_true_all, y_pred_all, labels=labels_sklearn)
        cm_df = pd.DataFrame(cm_sklearn, index=target_names, columns=target_names)
        cm_csv_path = OUT_DIR / "confusion_matrix_sklearn_iou_bg.csv"
        cm_df.to_csv(cm_csv_path)
        print(f"[SCIKIT] Confusion matrix saved to: {cm_csv_path}")

        # Optional heatmap
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, square=True)
            plt.title(f"Confusion Matrix (IoU≥{IOU_FOR_CONFMAT:.2f}, with background)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "confusion_matrix_sklearn_iou_bg.png", dpi=200)
            plt.close()
        except Exception as e:
            print(f"[WARN] Could not plot confusion matrix: {e}")

if __name__ == "__main__":
    main()
