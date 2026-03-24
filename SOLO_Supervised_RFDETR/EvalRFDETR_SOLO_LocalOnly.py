#!/usr/bin/env python
"""
Local RF-DETR evaluation script.

This script is intentionally local-only:
- No /work or UCloud path handling.
- No remote path rewrite logic.
- Assumes model checkpoint + COCO test json + image files are available locally.

Outputs:
- under: <output-dir>_<TargetName>/<YYYYMMDD-HHMMSS>/
- eval_summary.json
- predictions_coco.json
- iou_sweep_metrics.csv
- per_class_iou_sweep.csv
- per_class_metrics.csv
- confusion_matrix.json
- confusion_matrix.csv
- confusion_matrix.png
- threshold_metrics.csv
- pr_curve_overall.csv (if sklearn available)
- roc_curve_overall.csv (if sklearn available and both classes present)
- optional PNG plots + overlays/
- optional error_overlays/ containing missed GT and false-positive images
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ============================================================================
# USER PRESETS (EDIT THESE)
# ============================================================================
# Pick which preset to run by default.
# CLI can still override any field:
#   --preset leucocyte --checkpoint ... --test-json ... --output-dir ... --images-root ...
ACTIVE_PRESET = "leucocyte"

DEFAULT_OUTPUT_DIR = r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\EvaluationOutput"
# Prefix applied to relative file_name entries from COCO JSON, e.g.
# "Sample 15/Patches for Sample 15/....tif"
DEFAULT_IMAGES_ROOT = r"D:\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment_Cleaned"

# Overlay selection:
#   "all"  -> generate overlays for all processed test images
#   "25"   -> generate 25 randomly selected overlays
#   "0"    -> disable overlay generation
OVERLAY_SELECTION = "all"

# Error overlay selection:
#   "all"  -> generate error overlays for all images containing errors
#   "25"   -> generate 25 randomly selected error overlays
#   "0"    -> disable error overlay generation
ERROR_OVERLAY_SELECTION = "all"
INFERENCE_PROGRESS_EVERY = 10

EVAL_PRESETS = {
    "leucocyte": {
        "target_name": "Leucocyte",
        "checkpoint": r"D:\PHD\Results\Quality Assessment\General results\Lucocyte model default pretrained 40% AP@50\Leucocyte\HPO_Config_001\checkpoint_best_total.pth",
        "test_json": r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\SOLO_Supervised_RFDETR\Stat_Dataset\QA-2025v1_Leucocyte_OVR_V2_20260323-120328\test\_annotations.coco.json",
        "images_root": DEFAULT_IMAGES_ROOT,
        "model_class": "auto",
    },
    "epithelial": {
        "target_name": "Epithelial",
        "checkpoint": r"D:\PHD\Results\Quality Assessment\Epi+Leu for ESCMID Conference\first full Epi model no SSL\HPO_Config_003\checkpoint_best_total.pth",
        "test_json": r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\SOLO_Supervised_RFDETR\Stat_Dataset\QA-2025v1_SquamousEpithelialCell_OVR_V2_20260323-120604\test\_annotations.coco.json",
        "images_root": DEFAULT_IMAGES_ROOT,
        "model_class": "auto",
    },
}


def _optional_path(raw: str) -> Optional[Path]:
    raw = (raw or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _resolve_preset(name: str) -> Dict[str, str]:
    key = (name or "").strip().lower()
    if key not in EVAL_PRESETS:
        raise ValueError(
            f"Unknown preset {name!r}. Available presets: {', '.join(sorted(EVAL_PRESETS))}"
        )
    return EVAL_PRESETS[key]


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

try:
    import numpy as np
except Exception:
    np = None  # type: ignore[assignment]

try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]

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
    preset_name: str
    checkpoint: Path
    test_json: Path
    output_dir: Path
    output_root_dir: Path
    run_timestamp: str
    target_name: str
    images_root: Optional[Path]
    model_class: str
    model_resolution: Optional[int]
    score_floor: float
    score_threshold: float
    confmat_iou: float
    curve_iou: float
    iou_min: float
    iou_max: float
    iou_step: float
    threshold_points: int
    max_images: Optional[int]
    overlay_selection: str
    num_overlays: int
    error_overlay_selection: str
    num_error_overlays: int
    image_max_side: int
    seed: int
    progress_every: int
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


def patch_transformers_torch_compat() -> None:
    if torch is None:
        return

    dtype_aliases = {
        "uint16": "int16",
        "uint32": "int32",
        "uint64": "int64",
    }
    patched_dtypes: List[str] = []
    for missing_name, fallback_name in dtype_aliases.items():
        if hasattr(torch, missing_name):
            continue
        fallback_dtype = getattr(torch, fallback_name, None)
        if fallback_dtype is None:
            continue
        setattr(torch, missing_name, fallback_dtype)
        patched_dtypes.append(f"{missing_name}->{fallback_name}")
    if patched_dtypes:
        print(
            "[LOCAL EVAL][WARN] Added Torch dtype compatibility aliases for local eval: "
            + ", ".join(patched_dtypes)
        )

    try:
        import transformers.utils as tf_utils
        import transformers.utils.import_utils as tf_import_utils
    except Exception:
        return

    torch_version = str(torch.__version__)

    def _always_true() -> bool:
        return True

    def _torch_version() -> str:
        return torch_version

    for fn_name in ("is_torch_available", "get_torch_version"):
        fn = getattr(tf_import_utils, fn_name, None)
        if callable(fn) and hasattr(fn, "cache_clear"):
            try:
                fn.cache_clear()
            except Exception:
                pass

    tf_import_utils.is_torch_available = _always_true
    tf_import_utils.get_torch_version = _torch_version
    tf_utils.is_torch_available = _always_true
    tf_utils.get_torch_version = _torch_version

    if hasattr(tf_import_utils, "_torch_available"):
        tf_import_utils._torch_available = True
    if hasattr(tf_import_utils, "_torch_version"):
        tf_import_utils._torch_version = torch_version

    for key in list(sys.modules):
        if key in {
            "transformers.conversion_mapping",
            "transformers.core_model_loading",
            "transformers.modeling_utils",
            "transformers.integrations.accelerate",
        } or key.startswith("transformers.integrations.accelerate."):
            sys.modules.pop(key, None)


def build_env_mismatch_hint(exc: Exception) -> str:
    exc_text = str(exc)
    mismatch_tokens = (
        "PyTorch >= 2.4",
        "name 'nn' is not defined",
        "torch' has no attribute 'uint16'",
        "find_pruneable_heads_and_indices",
    )
    if not any(token in exc_text for token in mismatch_tokens):
        return ""

    try:
        transformers_version = importlib.metadata.version("transformers")
    except Exception:
        transformers_version = "unknown"
    torch_version = str(torch.__version__) if torch is not None else "missing"
    return (
        " Detected environment mismatch: "
        f"torch={torch_version}, transformers={transformers_version}. "
        "This local RF-DETR evaluator expects a Transformers 4.x API, while "
        "Transformers 5.x requires torch>=2.4 and removes symbols used by "
        "rfdetr_local. Recommended fix: use a compatible environment such as "
        "'transformers==4.57.2' with the current torch 2.1 stack, or upgrade "
        "torch and the RF-DETR/Transformers stack together."
    )


def ensure_rfdetr_import() -> None:
    try:
        import rfdetr  # noqa: F401
        return
    except Exception as installed_exc:
        project_root = Path(__file__).resolve().parents[1]
        local_pkg_dir = project_root / "rfdetr_local"
        init_py = local_pkg_dir / "__init__.py"
        if not init_py.exists():
            raise ImportError(
                f"Installed rfdetr import failed ({installed_exc}) and repo-local fallback "
                f"was not found at {local_pkg_dir}."
            ) from installed_exc

        project_root_str = str(project_root)
        local_pkg_dir_str = str(local_pkg_dir)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        if local_pkg_dir_str not in sys.path:
            sys.path.insert(0, local_pkg_dir_str)

        for key in list(sys.modules):
            if key == "rfdetr" or key.startswith("rfdetr."):
                sys.modules.pop(key, None)

        try:
            print(
                f"[LOCAL EVAL][WARN] Installed rfdetr import failed ({installed_exc}); "
                f"using repo-local fallback from {local_pkg_dir}."
            )
            patch_transformers_torch_compat()
            spec = importlib.util.spec_from_file_location(
                "rfdetr",
                init_py,
                submodule_search_locations=[str(local_pkg_dir)],
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create import spec for {init_py}")
            module = importlib.util.module_from_spec(spec)
            sys.modules["rfdetr"] = module
            spec.loader.exec_module(module)
        except Exception as local_exc:
            raise ImportError(
                f"Installed rfdetr import failed ({installed_exc}) and repo-local fallback "
                f"from {local_pkg_dir} also failed ({local_exc})."
                f"{build_env_mismatch_hint(local_exc)}"
            ) from local_exc


def normalize_target_name(raw: str) -> str:
    txt = (raw or "").strip()
    if not txt:
        return "Unknown"

    low = txt.lower()
    if any(k in low for k in ("leucocyte", "leukocyte", "leuco", "leuko", " wbc", "wbc ")):
        return "Leucocyte"
    if any(k in low for k in ("epithelial", "squamous")):
        return "Epithelial"
    if low in {"epi", "epithelial"}:
        return "Epithelial"
    if low in {"leu", "leucocyte", "leukocyte"}:
        return "Leucocyte"

    compact = re.sub(r"\s+", " ", txt)
    return compact or "Unknown"


def infer_target_name_from_coco_json(test_json: Path) -> Optional[str]:
    try:
        payload = json.loads(test_json.read_text(encoding="utf-8"))
    except Exception:
        return None

    categories = payload.get("categories", [])
    names = [str(c.get("name", "")).strip() for c in categories if isinstance(c, dict)]
    if not names:
        return None

    for name in names:
        normalized = normalize_target_name(name)
        if normalized in {"Epithelial", "Leucocyte"}:
            return normalized

    if len(names) == 1:
        return normalize_target_name(names[0])
    return None


def infer_target_name_from_path(path: Path) -> Optional[str]:
    text = str(path).lower()
    has_epi = any(tok in text for tok in ("epithelial", "squamous", "_epi", "-epi", "\\epi\\", "/epi/"))
    has_leu = any(tok in text for tok in ("leucocyte", "leukocyte", "leuco", "leuko", "_leu", "-leu", "\\leu\\", "/leu/"))
    if has_epi and not has_leu:
        return "Epithelial"
    if has_leu and not has_epi:
        return "Leucocyte"
    return None


def sanitize_target_suffix(target_name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", normalize_target_name(target_name))
    return cleaned or "Unknown"


def resolve_target_name(
    args: argparse.Namespace,
    checkpoint: Path,
    test_json: Path,
    preset_target_name: str = "",
) -> str:
    explicit_name = (args.target_name or "").strip()
    if explicit_name:
        return normalize_target_name(explicit_name)

    if args.target == "epithelial":
        return "Epithelial"
    if args.target == "leucocyte":
        return "Leucocyte"

    if preset_target_name:
        return normalize_target_name(preset_target_name)

    coco_target = infer_target_name_from_coco_json(test_json)
    if coco_target:
        return coco_target

    path_target = infer_target_name_from_path(test_json) or infer_target_name_from_path(checkpoint)
    if path_target:
        return path_target

    return "Unknown"


def build_output_dir(base_output_dir: Path, target_name: str) -> Tuple[Path, Path, str]:
    suffix = sanitize_target_suffix(target_name)
    base_name = base_output_dir.name
    suffix_tag = f"_{suffix.lower()}"
    if base_name.lower().endswith(suffix_tag):
        root_dir = base_output_dir.parent / base_name
    else:
        root_dir = base_output_dir.parent / f"{base_name}_{suffix}"
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = root_dir / run_stamp
    counter = 2
    while out_dir.exists():
        out_dir = root_dir / f"{run_stamp}_{counter:02d}"
        counter += 1
    return out_dir, root_dir, out_dir.name


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
    meta_candidates = [
        model_meta_root / "run_meta" / "model_architecture.json",
        model_meta_root / "rfdetr_run" / "run_meta" / "model_architecture.json",
    ]
    for meta_model in meta_candidates:
        if not meta_model.exists():
            continue
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


def infer_model_resolution(checkpoint: Path) -> Optional[int]:
    # Prefer lightweight sidecar metadata from training.
    sidecars = [
        checkpoint.parent / "run_meta" / "train_kwargs.json",
        checkpoint.parent.parent / "run_meta" / "train_kwargs.json",
        checkpoint.parent / "rfdetr_run" / "run_meta" / "train_kwargs.json",
    ]
    for p in sidecars:
        if not p.exists():
            continue
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
            val = js.get("resolution")
            if val is not None:
                return int(val)
        except Exception:
            pass

    # Fallback: inspect checkpoint args.
    try:
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)  # type: ignore[union-attr]
        args = ckpt.get("args", None)
        if args is None:
            return None
        if isinstance(args, dict):
            val = args.get("resolution")
            return int(val) if val is not None else None
        val = getattr(args, "resolution", None)
        return int(val) if val is not None else None
    except Exception:
        return None


def load_model(model_class: str, checkpoint: Path, resolution: Optional[int]):
    ensure_rfdetr_import()
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
    model = name_to_cls[model_class](**kwargs)
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
                for kw in ({"threshold": float(score_floor)}, {}):
                    try:
                        out = fn(inp, **kw)
                        break
                    except TypeError:
                        out = None
                    except Exception:
                        out = None
                if out is not None:
                    break
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


def match_errors_for_sample(sample: Dict[str, Any], score_threshold: float, iou_thr: float) -> Dict[str, Any]:
    gt_boxes = sample["gt_boxes"]
    gt_cls = sample["gt_cls"]

    keep = sample["pred_scores"] >= score_threshold
    pred_boxes = sample["pred_boxes"][keep]
    pred_cls = sample["pred_cls"][keep]
    pred_scores = sample["pred_scores"][keep]

    ious = iou_matrix(gt_boxes, pred_boxes)
    valid = gt_cls[:, None] == pred_cls[None, :] if len(gt_boxes) and len(pred_boxes) else None
    matches = greedy_match(ious, iou_thr=iou_thr, valid_pairs=valid)

    matched_gt = {gi for gi, _ in matches}
    matched_pr = {pj for _, pj in matches}
    false_negative_idx = [gi for gi in range(len(gt_cls)) if gi not in matched_gt]
    false_positive_idx = [pj for pj in range(len(pred_cls)) if pj not in matched_pr]

    return {
        "pred_boxes_kept": pred_boxes,
        "pred_cls_kept": pred_cls,
        "pred_scores_kept": pred_scores,
        "matches": matches,
        "false_negative_idx": false_negative_idx,
        "false_positive_idx": false_positive_idx,
    }


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


def save_confusion_matrix_plot(
    path: Path,
    labels: List[str],
    cm: np.ndarray,
    score_threshold: float,
    iou_threshold: float,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[WARN] matplotlib unavailable; skipping confusion matrix PNG.")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(cm, dtype=np.int64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(np.float64),
        np.where(row_sums == 0, 1, row_sums),
        out=np.zeros_like(cm, dtype=np.float64),
    )

    n_rows, n_cols = cm.shape
    fig_w = max(6.5, 1.35 * n_cols + 2.0)
    fig_h = max(5.5, 1.10 * n_rows + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized fraction", rotation=90)

    ax.set(
        xticks=np.arange(n_cols),
        yticks=np.arange(n_rows),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="True class",
        title=(
            "Confusion Matrix\n"
            f"Score >= {score_threshold:.2f}, IoU >= {iou_threshold:.2f}"
        ),
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = 0.5
    for i in range(n_rows):
        for j in range(n_cols):
            count = int(cm[i, j])
            frac = float(cm_norm[i, j])
            if row_sums[i, 0] > 0:
                text = f"{count}\n{frac * 100.0:.1f}%"
            else:
                text = str(count)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if frac > threshold else "black",
                fontsize=10,
                fontweight="bold" if count > 0 else "normal",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def _measure_text(draw: Any, text: str, font: Any) -> Tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return int(right - left), int(bottom - top)
    if hasattr(draw, "textsize"):
        width, height = draw.textsize(text, font=font)
        return int(width), int(height)
    return (max(1, len(text) * 7), 14)


def _rects_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], pad: float = 0.0) -> bool:
    return not (
        (a[2] + pad) < b[0]
        or (b[2] + pad) < a[0]
        or (a[3] + pad) < b[1]
        or (b[3] + pad) < a[1]
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


def _choose_text_position(
    box: Tuple[float, float, float, float],
    text_size: Tuple[int, int],
    image_size: Tuple[int, int],
    occupied_rects: Sequence[Tuple[float, float, float, float]],
) -> Tuple[float, float]:
    img_w, img_h = image_size
    x1, y1, x2, y2 = box
    text_w, text_h = text_size
    margin = 3.0
    box_rect = (x1, y1, x2, y2)

    candidates = [
        (_clamp(x1, 0.0, max(0.0, img_w - text_w)), y1 - text_h - margin),
        (_clamp(x2 - text_w, 0.0, max(0.0, img_w - text_w)), y1 - text_h - margin),
        (_clamp(x1, 0.0, max(0.0, img_w - text_w)), y2 + margin),
        (_clamp(x2 - text_w, 0.0, max(0.0, img_w - text_w)), y2 + margin),
        (x2 + margin, _clamp(y1, 0.0, max(0.0, img_h - text_h))),
        (x1 - text_w - margin, _clamp(y1, 0.0, max(0.0, img_h - text_h))),
    ]

    for cand_x, cand_y in candidates:
        text_rect = (cand_x, cand_y, cand_x + text_w, cand_y + text_h)
        if cand_x < 0 or cand_y < 0 or text_rect[2] > img_w or text_rect[3] > img_h:
            continue
        if _rects_overlap(text_rect, box_rect, pad=1.0):
            continue
        if any(_rects_overlap(text_rect, other, pad=2.0) for other in occupied_rects):
            continue
        return cand_x, cand_y

    fallback_x = _clamp(x1, 0.0, max(0.0, img_w - text_w))
    fallback_y = _clamp(max(0.0, y1 - text_h - margin), 0.0, max(0.0, img_h - text_h))
    return fallback_x, fallback_y


def _draw_outlined_text(draw: Any, xy: Tuple[float, float], text: str, fill: Tuple[int, int, int], font: Any) -> None:
    x, y = xy
    outline_fill = (255, 255, 255)
    for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        draw.text((x + dx, y + dy), text, fill=outline_fill, font=font)
    draw.text((x, y), text, fill=fill, font=font)


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
    del gt_names, pred_names

    gt_color = (0, 200, 0)
    pred_color = (220, 30, 30)
    gt_box_width = 3
    pred_box_width = 2
    img = Image.open(img_path).convert("RGB")
    scale = min(1.0, float(image_max_side) / float(max(img.size)))
    if scale < 0.999:
        new_wh = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_wh, Image.BILINEAR)

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default() if ImageFont is not None else None
    occupied_text_rects: List[Tuple[float, float, float, float]] = []

    def sc(box: np.ndarray) -> Tuple[float, float, float, float]:
        return (
            float(box[0] * scale),
            float(box[1] * scale),
            float(box[2] * scale),
            float(box[3] * scale),
        )

    for box in gt_boxes:
        x1, y1, x2, y2 = sc(box)
        draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=gt_box_width)

    for box, score in zip(pred_boxes, pred_scores):
        x1, y1, x2, y2 = sc(box)
        draw.rectangle([x1, y1, x2, y2], outline=pred_color, width=pred_box_width)
        score_text = f"prob:{float(score):.2f}"
        text_w, text_h = _measure_text(draw, score_text, font)
        text_x, text_y = _choose_text_position(
            (x1, y1, x2, y2),
            (text_w, text_h),
            img.size,
            occupied_text_rects,
        )
        _draw_outlined_text(draw, (text_x, text_y), score_text, pred_color, font)
        occupied_text_rects.append((text_x, text_y, text_x + text_w, text_y + text_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def draw_error_overlay(
    img_path: Path,
    missed_gt_boxes: np.ndarray,
    missed_gt_names: List[str],
    false_pos_boxes: np.ndarray,
    false_pos_names: List[str],
    false_pos_scores: np.ndarray,
    out_path: Path,
    image_max_side: int,
) -> None:
    del missed_gt_names, false_pos_names

    missed_color = (245, 170, 0)
    false_pos_color = (220, 30, 30)
    missed_box_width = 3
    false_pos_box_width = 2
    img = Image.open(img_path).convert("RGB")
    scale = min(1.0, float(image_max_side) / float(max(img.size)))
    if scale < 0.999:
        new_wh = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_wh, Image.BILINEAR)

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default() if ImageFont is not None else None
    occupied_text_rects: List[Tuple[float, float, float, float]] = []

    def sc(box: np.ndarray) -> Tuple[float, float, float, float]:
        return (
            float(box[0] * scale),
            float(box[1] * scale),
            float(box[2] * scale),
            float(box[3] * scale),
        )

    for box in missed_gt_boxes:
        x1, y1, x2, y2 = sc(box)
        draw.rectangle([x1, y1, x2, y2], outline=missed_color, width=missed_box_width)

    for box, score in zip(false_pos_boxes, false_pos_scores):
        x1, y1, x2, y2 = sc(box)
        draw.rectangle([x1, y1, x2, y2], outline=false_pos_color, width=false_pos_box_width)
        score_text = f"prob:{float(score):.2f}"
        text_w, text_h = _measure_text(draw, score_text, font)
        text_x, text_y = _choose_text_position(
            (x1, y1, x2, y2),
            (text_w, text_h),
            img.size,
            occupied_text_rects,
        )
        _draw_outlined_text(draw, (text_x, text_y), score_text, false_pos_color, font)
        occupied_text_rects.append((text_x, text_y, text_x + text_w, text_y + text_h))

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
    eval_start_t = time.perf_counter()

    if not cfg.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {cfg.checkpoint}")
    if not cfg.test_json.exists():
        raise FileNotFoundError(f"test_json not found: {cfg.test_json}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[LOCAL EVAL] Preset={cfg.preset_name}")
    print(f"[LOCAL EVAL] Target={cfg.target_name}")
    print(f"[LOCAL EVAL] Model checkpoint={cfg.checkpoint}")
    print(f"[LOCAL EVAL] Test JSON={cfg.test_json}")
    print(f"[LOCAL EVAL] Output directory={cfg.output_dir}")
    print(f"[LOCAL EVAL] Overlay selection={cfg.overlay_selection}")
    print(f"[LOCAL EVAL] Error overlay selection={cfg.error_overlay_selection}")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    coco = COCO(str(cfg.test_json))
    cat_id_to_name, id_to_idx, idx_to_id = coco_categories(coco)
    n_classes = len(id_to_idx)
    labels = [cat_id_to_name[idx_to_id[i]] for i in range(n_classes)]

    model = load_model(cfg.model_class, cfg.checkpoint, cfg.model_resolution)
    if cfg.model_resolution is not None:
        print(f"[LOCAL EVAL] Using model resolution={cfg.model_resolution} (from training metadata/checkpoint)")
    else:
        print("[LOCAL EVAL][WARN] Could not infer training resolution; using library default resolution.")

    img_ids = coco.getImgIds()
    if cfg.max_images is not None:
        img_ids = img_ids[: cfg.max_images]
    total_requested_images = len(img_ids)
    img_id_set = set(int(i) for i in img_ids)
    total_gt_objects_in_requested_testset = int(
        sum(
            1
            for ann in coco.dataset.get("annotations", [])
            if int(ann.get("image_id", -1)) in img_id_set
            and int(ann.get("iscrowd", 0)) == 0
            and int(ann.get("category_id", -1)) in id_to_idx
        )
    )

    try:
        from tqdm import tqdm

        iterator = tqdm(img_ids, desc="Infer", unit="img")
    except Exception:
        iterator = img_ids

    preds_coco: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    inference_seconds_total = 0.0
    inference_images_timed = 0

    for idx, img_id in enumerate(iterator, start=1):
        if cfg.progress_every > 0 and (
            idx == 1
            or idx % cfg.progress_every == 0
            or idx == total_requested_images
        ):
            pct = (idx / total_requested_images) if total_requested_images > 0 else 1.0
            print(
                f"[LOCAL EVAL] Inference progress: {idx}/{total_requested_images} "
                f"({pct:.1%})"
            )
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
        infer_t0 = time.perf_counter()
        pred_boxes, pred_scores, pred_labels = predict_one_image(model, image, cfg.score_floor)
        inference_seconds_total += (time.perf_counter() - infer_t0)
        inference_images_timed += 1
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
    confusion_matrix_png_created = False
    if not cfg.no_plots:
        confusion_matrix_png_created = save_confusion_matrix_plot(
            cfg.output_dir / "confusion_matrix.png",
            cm_labels,
            cm,
            cfg.score_threshold,
            cfg.confmat_iou,
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

    per_image_count_rows: List[Dict[str, Any]] = []
    per_image_error_rows: List[Dict[str, Any]] = []
    error_overlay_samples: List[Dict[str, Any]] = []
    total_false_negatives = 0
    total_false_positives = 0
    for s in samples:
        keep = s["pred_scores"] >= cfg.score_threshold
        error_info = match_errors_for_sample(s, cfg.score_threshold, cfg.confmat_iou)
        false_negative_idx = error_info["false_negative_idx"]
        false_positive_idx = error_info["false_positive_idx"]
        missed_gt_boxes = s["gt_boxes"][false_negative_idx]
        missed_gt_cls = s["gt_cls"][false_negative_idx]
        false_pos_boxes = error_info["pred_boxes_kept"][false_positive_idx]
        false_pos_cls = error_info["pred_cls_kept"][false_positive_idx]
        false_pos_scores = error_info["pred_scores_kept"][false_positive_idx]
        false_negative_count = int(len(false_negative_idx))
        false_positive_count = int(len(false_positive_idx))
        total_false_negatives += false_negative_count
        total_false_positives += false_positive_count

        per_image_count_rows.append(
            {
                "image_name": Path(s["img_path"]).name,
                "image_path": s["img_path"],
                "gt_objects": int(len(s["gt_boxes"])),
                "pred_objects_at_threshold": int(np.sum(keep)),
                "score_threshold": float(cfg.score_threshold),
            }
        )
        per_image_error_rows.append(
            {
                "image_name": Path(s["img_path"]).name,
                "image_path": s["img_path"],
                "false_negatives": false_negative_count,
                "false_positives": false_positive_count,
                "total_errors": false_negative_count + false_positive_count,
                "score_threshold": float(cfg.score_threshold),
                "iou_threshold": float(cfg.confmat_iou),
            }
        )
        if false_negative_count > 0 or false_positive_count > 0:
            error_overlay_samples.append(
                {
                    "img_path": s["img_path"],
                    "missed_gt_boxes": missed_gt_boxes,
                    "missed_gt_names": [labels[int(c)] for c in missed_gt_cls],
                    "false_pos_boxes": false_pos_boxes,
                    "false_pos_names": [labels[int(c)] for c in false_pos_cls],
                    "false_pos_scores": false_pos_scores,
                    "false_negatives": false_negative_count,
                    "false_positives": false_positive_count,
                }
            )
    write_csv(
        cfg.output_dir / "per_image_object_counts.csv",
        ["image_name", "image_path", "gt_objects", "pred_objects_at_threshold", "score_threshold"],
        per_image_count_rows,
    )
    write_csv(
        cfg.output_dir / "per_image_error_counts.csv",
        ["image_name", "image_path", "false_negatives", "false_positives", "total_errors", "score_threshold", "iou_threshold"],
        per_image_error_rows,
    )

    if cfg.num_overlays != 0:
        chosen = list(samples)
        random.shuffle(chosen)
        if cfg.num_overlays > 0:
            chosen = chosen[: cfg.num_overlays]
        used_overlay_names: set[str] = set()
        for i, s in enumerate(chosen, start=1):
            keep = s["pred_scores"] >= cfg.score_threshold
            pred_boxes = s["pred_boxes"][keep]
            pred_cls = s["pred_cls"][keep]
            pred_scores = s["pred_scores"][keep]
            gt_names = [labels[int(c)] for c in s["gt_cls"]]
            pred_names = [labels[int(c)] for c in pred_cls]
            gt_count = int(len(s["gt_boxes"]))
            pred_count = int(len(pred_boxes))
            img_stem = Path(s["img_path"]).stem
            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", img_stem).strip("._")
            if not safe_stem:
                safe_stem = f"image_{i:02d}"
            base_name = f"Overlay_{safe_stem}_GT{gt_count}_Pred{pred_count}"
            overlay_name = f"{base_name}.jpg"
            suffix = 2
            while overlay_name in used_overlay_names:
                overlay_name = f"{base_name}_{suffix}.jpg"
                suffix += 1
            used_overlay_names.add(overlay_name)
            draw_overlay(
                img_path=Path(s["img_path"]),
                gt_boxes=s["gt_boxes"],
                gt_names=gt_names,
                pred_boxes=pred_boxes,
                pred_names=pred_names,
                pred_scores=pred_scores,
                out_path=cfg.output_dir / "overlays" / overlay_name,
                image_max_side=cfg.image_max_side,
            )

    if cfg.num_error_overlays != 0:
        chosen_error_samples = list(error_overlay_samples)
        random.shuffle(chosen_error_samples)
        if cfg.num_error_overlays > 0:
            chosen_error_samples = chosen_error_samples[: cfg.num_error_overlays]
        used_error_overlay_names: set[str] = set()
        for i, s in enumerate(chosen_error_samples, start=1):
            img_stem = Path(s["img_path"]).stem
            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", img_stem).strip("._")
            if not safe_stem:
                safe_stem = f"error_image_{i:02d}"
            base_name = (
                f"Errors_{safe_stem}_FN{s['false_negatives']}_FP{s['false_positives']}"
            )
            overlay_name = f"{base_name}.jpg"
            suffix = 2
            while overlay_name in used_error_overlay_names:
                overlay_name = f"{base_name}_{suffix}.jpg"
                suffix += 1
            used_error_overlay_names.add(overlay_name)
            draw_error_overlay(
                img_path=Path(s["img_path"]),
                missed_gt_boxes=s["missed_gt_boxes"],
                missed_gt_names=s["missed_gt_names"],
                false_pos_boxes=s["false_pos_boxes"],
                false_pos_names=s["false_pos_names"],
                false_pos_scores=s["false_pos_scores"],
                out_path=cfg.output_dir / "error_overlays" / overlay_name,
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
        "preset_name": cfg.preset_name,
        "target_name": cfg.target_name,
        "run_timestamp": cfg.run_timestamp,
        "checkpoint": str(cfg.checkpoint),
        "test_json": str(cfg.test_json),
        "output_dir": str(cfg.output_dir),
        "output_root_dir": str(cfg.output_root_dir),
        "images_root": str(cfg.images_root) if cfg.images_root is not None else None,
        "model_class": cfg.model_class,
        "model_resolution": cfg.model_resolution,
        "used_model": {
            "checkpoint_path": str(cfg.checkpoint),
            "checkpoint_file": cfg.checkpoint.name,
            "model_class": cfg.model_class,
            "model_resolution": cfg.model_resolution,
        },
        "images_total_requested": len(img_ids),
        "images_processed": len(processed_ids),
        "images_missing": len(missing),
        "total_gt_objects_in_requested_testset": total_gt_objects_in_requested_testset,
        "total_gt_objects_in_processed_images": int(sum(len(s["gt_boxes"]) for s in samples)),
        "missing_images": missing,
        "score_floor": cfg.score_floor,
        "score_threshold": cfg.score_threshold,
        "confmat_iou": cfg.confmat_iou,
        "curve_iou": cfg.curve_iou,
        "overlay_selection": cfg.overlay_selection,
        "error_overlay_selection": cfg.error_overlay_selection,
        "iou_sweep": {"min": cfg.iou_min, "max": cfg.iou_max, "step": cfg.iou_step, "rows": iou_rows},
        "coco_standard": coco_std_summary,
        "best_f1": best_f1_row,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "roc_note": "ROC/PR are computed from IoU-matched detection samples.",
        "confusion_matrix_png": str(cfg.output_dir / "confusion_matrix.png") if confusion_matrix_png_created else None,
        "per_image_object_counts_csv": str(cfg.output_dir / "per_image_object_counts.csv"),
        "per_image_error_counts_csv": str(cfg.output_dir / "per_image_error_counts.csv"),
        "error_summary": {
            "images_with_errors": int(len(error_overlay_samples)),
            "total_false_negatives": int(total_false_negatives),
            "total_false_positives": int(total_false_positives),
            "error_iou_threshold": float(cfg.confmat_iou),
            "error_score_threshold": float(cfg.score_threshold),
        },
        "timing": {
            "inference_seconds_total": float(inference_seconds_total),
            "inference_images_timed": int(inference_images_timed),
            "inference_avg_ms_per_image": float((inference_seconds_total / inference_images_timed) * 1000.0) if inference_images_timed > 0 else None,
            "inference_images_per_second": float(inference_images_timed / inference_seconds_total) if inference_seconds_total > 0 else None,
            "evaluation_wall_clock_seconds": float(time.perf_counter() - eval_start_t),
        },
    }
    json_dump(cfg.output_dir / "eval_summary.json", summary)

    print("\n[LOCAL EVAL] Done")
    print(f"[LOCAL EVAL] AP@50:95={coco_std_summary['AP@50:95']:.4f} AP@50={coco_std_summary['AP@50']:.4f} AP@75={coco_std_summary['AP@75']:.4f}")
    if inference_images_timed > 0 and inference_seconds_total > 0:
        print(
            f"[LOCAL EVAL] Inference: {inference_seconds_total:.3f}s total over {inference_images_timed} images "
            f"({(inference_seconds_total / inference_images_timed) * 1000.0:.2f} ms/img, "
            f"{(inference_images_timed / inference_seconds_total):.2f} img/s)"
        )
    if best_f1_row is not None:
        print(f"[LOCAL EVAL] Best F1={float(best_f1_row['f1']):.4f} at threshold={float(best_f1_row['threshold']):.4f}")
    print(
        f"[LOCAL EVAL] Errors at score>={cfg.score_threshold:.2f}, IoU>={cfg.confmat_iou:.2f}: "
        f"FN={total_false_negatives} FP={total_false_positives} images_with_errors={len(error_overlay_samples)}"
    )
    print(f"[LOCAL EVAL] Summary -> {(cfg.output_dir / 'eval_summary.json').resolve()}")


def build_config(args: argparse.Namespace) -> LocalEvalConfig:
    preset_name = (args.preset or ACTIVE_PRESET).strip().lower()
    preset = _resolve_preset(preset_name)

    checkpoint_raw = args.checkpoint or _optional_path(preset.get("checkpoint", ""))
    test_json_raw = args.test_json or _optional_path(preset.get("test_json", ""))
    output_dir_raw = args.output_dir or _optional_path(preset.get("output_dir", "")) or _optional_path(DEFAULT_OUTPUT_DIR)
    images_root_raw = args.images_root or _optional_path(preset.get("images_root", ""))

    if checkpoint_raw is None:
        raise ValueError(
            f"No checkpoint configured for preset '{preset_name}'. "
            "Edit EVAL_PRESETS or pass --checkpoint."
        )
    if test_json_raw is None:
        raise ValueError(
            f"No test JSON configured for preset '{preset_name}'. "
            "Edit EVAL_PRESETS or pass --test-json."
        )
    if output_dir_raw is None:
        raise ValueError("No output directory configured. Edit DEFAULT_OUTPUT_DIR or pass --output-dir.")

    checkpoint = checkpoint_raw.resolve()
    test_json = test_json_raw.resolve()
    base_output_dir = output_dir_raw.resolve()
    images_root = images_root_raw.resolve() if images_root_raw is not None else None
    model_meta_root = checkpoint.parent.parent if checkpoint.parent.name.lower() == "rfdetr_run" else checkpoint.parent

    model_resolution = infer_model_resolution(checkpoint)
    model_class_choice = args.model_class
    if model_class_choice == "auto":
        model_class_choice = str(preset.get("model_class", "auto"))
    model_class = infer_model_class(model_meta_root, checkpoint) if model_class_choice == "auto" else model_class_choice
    target_name = resolve_target_name(
        args,
        checkpoint,
        test_json,
        preset_target_name=str(preset.get("target_name", "")),
    )
    output_dir, output_root_dir, run_timestamp = build_output_dir(base_output_dir, target_name)
    overlay_selection = str(args.overlay_selection or OVERLAY_SELECTION).strip()
    num_overlays = parse_overlay_selection(overlay_selection)
    error_overlay_selection = str(args.error_overlay_selection or ERROR_OVERLAY_SELECTION).strip()
    num_error_overlays = parse_overlay_selection(error_overlay_selection)

    if args.iou_step <= 0:
        raise ValueError("--iou-step must be > 0")
    if args.iou_min <= 0 or args.iou_max > 1.0 or args.iou_min >= args.iou_max:
        raise ValueError("Need 0 < --iou-min < --iou-max <= 1.0")
    if args.threshold_points < 2:
        raise ValueError("--threshold-points must be >= 2")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be >= 0")

    return LocalEvalConfig(
        preset_name=preset_name,
        checkpoint=checkpoint,
        test_json=test_json,
        output_dir=output_dir,
        output_root_dir=output_root_dir,
        run_timestamp=run_timestamp,
        target_name=target_name,
        images_root=images_root,
        model_class=model_class,
        model_resolution=model_resolution,
        score_floor=float(args.score_floor),
        score_threshold=float(args.score_threshold),
        confmat_iou=float(args.confmat_iou),
        curve_iou=float(args.curve_iou),
        iou_min=float(args.iou_min),
        iou_max=float(args.iou_max),
        iou_step=float(args.iou_step),
        threshold_points=int(args.threshold_points),
        max_images=args.max_images,
        overlay_selection=overlay_selection,
        num_overlays=num_overlays,
        error_overlay_selection=error_overlay_selection,
        num_error_overlays=num_error_overlays,
        image_max_side=int(args.image_max_side),
        seed=int(args.seed),
        progress_every=int(args.progress_every),
        skip_missing_images=bool(args.skip_missing_images),
        no_plots=bool(args.no_plots),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local-only RF-DETR evaluator")
    p.add_argument(
        "--preset",
        type=str,
        default=ACTIVE_PRESET,
        choices=sorted(EVAL_PRESETS),
        help="Named evaluation preset from EVAL_PRESETS at the top of the file.",
    )
    p.add_argument("--checkpoint", type=Path, default=None, help="Path to model checkpoint file.")
    p.add_argument("--test-json", type=Path, default=None, help="Path to COCO test annotations JSON.")
    p.add_argument("--output-dir", type=Path, default=None, help="Directory to write evaluation outputs.")
    p.add_argument("--images-root", type=Path, default=None, help="Prefix root for relative image file_name entries.")
    p.add_argument(
        "--target",
        type=str,
        default="auto",
        choices=["auto", "epithelial", "leucocyte"],
        help="Evaluation target name used for output folder naming.",
    )
    p.add_argument(
        "--target-name",
        type=str,
        default="",
        help="Optional custom target display name; overrides --target when provided.",
    )
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
    p.add_argument(
        "--overlay-selection",
        type=str,
        default=OVERLAY_SELECTION,
        help="Overlay output selection: 'all', '0', or a positive integer for a random subset.",
    )
    p.add_argument(
        "--error-overlay-selection",
        type=str,
        default=ERROR_OVERLAY_SELECTION,
        help="Error overlay output selection: 'all', '0', or a positive integer for a random subset of images with errors.",
    )
    p.add_argument("--image-max-side", type=int, default=1600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--progress-every",
        type=int,
        default=INFERENCE_PROGRESS_EVERY,
        help="Print inference progress every N processed images. Use 0 to disable.",
    )

    p.add_argument("--skip-missing-images", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = build_config(args)
    run_local_eval(cfg)


if __name__ == "__main__":
    main()
