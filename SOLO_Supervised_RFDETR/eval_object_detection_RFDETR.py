#!/usr/bin/env python
"""
eval_object_detection_RFDETR.py

Run standard RF-DETR object-detection inference on a COCO test split and export
common detection metrics:
- COCO AP/AR (standard AP@50:95, AP@50, AP@75)
- IoU-swept AP/AR from 0.10 to 0.95 (step configurable)
- Confusion matrix with background class
- Threshold sweep (precision/recall/F1 + FP/image)
- Per-image count errors at the locked class-specific thresholds
- PR and ROC-style curves from IoU-matched detections
- Optional overlay images

Optional PyCharm workflow:
- edit the PYCHARM_* config block near the top of this file
- run the script with no command-line arguments
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfdetr_model_registry import (
    SUPPORTED_RFDETR_MODEL_NAME_SET,
    infer_rfdetr_model_name_from_checkpoint_name,
    instantiate_rfdetr_model,
    supported_rfdetr_model_names,
)

# -----------------------------------------------------------------------------
# Optional top-level config for running directly from PyCharm
# -----------------------------------------------------------------------------
# Enabled so running this script with no command-line arguments evaluates the
# final HPO-selected object detector on the held-out COCO test split.
PYCHARM_USE_TOP_LEVEL_CONFIG = True

DEFAULT_CLASS_SCORE_THRESHOLDS: Dict[str, float] = {
    "Leucocyte": 0.36,
    "Squamous Epithelial Cell": 0.35,
}
DEFAULT_CLASS_SCORE_THRESHOLDS_TEXT = ";".join(
    f"{name}={threshold}" for name, threshold in DEFAULT_CLASS_SCORE_THRESHOLDS.items()
)

PYCHARM_EVALUATE = {
    "run_dir": r"E:\PHD\Results\Quality Assessment\FINAL_B200\session_20260618_113853\TwoClass\HPO_Config_009",
    "checkpoint": r"E:\PHD\Results\Quality Assessment\FINAL_B200\session_20260618_113853\TwoClass\HPO_Config_009\checkpoint_best_ema.pth",
    "test_json": r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\SOLO_Supervised_RFDETR\Stat_Dataset\QA-2025v1_TwoClass_OVR_V2_20260618-101346\test\_annotations.coco.json",
    "output_dir": r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\EvaluationOutput\TwoClass_HPO009_EMA_TestEval",
    "model_class": "auto",
    "score_floor": 0.001,
    "score_threshold": 0.001,
    "class_score_thresholds": DEFAULT_CLASS_SCORE_THRESHOLDS_TEXT,
    "confmat_iou": 0.50,
    "curve_iou": 0.50,
    "iou_min": 0.10,
    "iou_max": 0.95,
    "iou_step": 0.05,
    "threshold_points": 51,
    "max_images": None,
    "num_overlays": 25,

    "image_max_side": 1600,
    "seed": 42,
    "path_rewrite": "",
    "images_root": r"E:\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment_Cleaned",
    "skip_missing_images": False,
    "no_plots": False,
}

# Optional imports are dependency-gated so argument parsing still works when the
# active environment is missing evaluation packages.
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
# RF-DETR import compatibility
# -----------------------------------------------------------------------------

def patch_transformers_torch_compat() -> None:
    if torch is None:
        return

    dtype_aliases = {
        "uint16": "int16",
        "uint32": "int32",
        "uint64": "int64",
    }
    for missing_name, fallback_name in dtype_aliases.items():
        if hasattr(torch, missing_name):
            continue
        fallback_dtype = getattr(torch, fallback_name, None)
        if fallback_dtype is not None:
            setattr(torch, missing_name, fallback_dtype)

    try:
        import transformers
        import transformers.utils as tf_utils
        import transformers.utils.backbone_utils as tf_backbone_utils
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

    for attr_name in ("BackboneConfigMixin", "BackboneMixin"):
        if attr_name not in transformers.__dict__ and hasattr(tf_backbone_utils, attr_name):
            setattr(transformers, attr_name, getattr(tf_backbone_utils, attr_name))

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


def _patch_transformers_pruning_compat() -> None:
    try:
        import transformers.pytorch_utils as pytorch_utils
    except Exception:
        return

    if torch is None or hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        return

    def find_pruneable_heads_and_indices(
        heads: set[int],
        n_heads: int,
        head_size: int,
        already_pruned_heads: set[int],
    ) -> tuple[set[int], Any]:
        heads = set(heads) - set(already_pruned_heads)
        mask = torch.ones(n_heads, head_size)
        for head in heads:
            head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def _patch_transformers_backbone_compat() -> None:
    try:
        import transformers.backbone_utils as backbone_utils
        import transformers.utils.backbone_utils as legacy_backbone_utils
    except Exception:
        return

    def get_aligned_output_features_output_indices(
        out_features: list[str] | tuple[str, ...] | None = None,
        out_indices: list[int] | tuple[int, ...] | None = None,
        stage_names: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[list[str], list[int]]:
        stage_names = list(stage_names or [])
        if not stage_names:
            return list(out_features or []), [int(value) for value in (out_indices or [])]
        if out_features is None and out_indices is None:
            return [stage_names[-1]], [len(stage_names) - 1]
        if out_features is None:
            resolved_indices = [int(value) for value in out_indices or []]
            return [stage_names[idx] for idx in resolved_indices], resolved_indices
        if out_indices is None:
            feature_to_index = {name: idx for idx, name in enumerate(stage_names)}
            resolved_features = [str(value) for value in out_features]
            return resolved_features, [feature_to_index[name] for name in resolved_features]
        return [str(value) for value in out_features], [int(value) for value in out_indices]

    for module in (backbone_utils, legacy_backbone_utils):
        if not hasattr(module, "get_aligned_output_features_output_indices"):
            module.get_aligned_output_features_output_indices = get_aligned_output_features_output_indices


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
    torch_version = str(torch.__version__) if torch is not None else "missing"
    return (
        " Detected environment mismatch: "
        f"torch={torch_version}. "
        "Use a compatible torch/transformers/RF-DETR environment for this repository."
    )


def ensure_rfdetr_import() -> None:
    patch_transformers_torch_compat()
    _patch_transformers_pruning_compat()
    _patch_transformers_backbone_compat()
    try:
        import rfdetr  # noqa: F401
        return
    except Exception as installed_exc:
        local_pkg_dir = PROJECT_ROOT / "rfdetr_local"
        init_py = local_pkg_dir / "__init__.py"
        if not init_py.exists():
            raise ImportError(
                f"Installed rfdetr import failed ({installed_exc}) and repo-local fallback "
                f"was not found at {local_pkg_dir}."
            ) from installed_exc

        project_root_str = str(PROJECT_ROOT)
        local_pkg_dir_str = str(local_pkg_dir)
        if project_root_str not in sys.path:
            sys.path.insert(0, project_root_str)
        if local_pkg_dir_str not in sys.path:
            sys.path.insert(0, local_pkg_dir_str)

        for key in list(sys.modules):
            if key == "rfdetr" or key.startswith("rfdetr."):
                sys.modules.pop(key, None)

        try:
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


# -----------------------------------------------------------------------------
# Shared path helpers
# -----------------------------------------------------------------------------

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
    if not text:
        return
    argv.extend([flag, text])


def build_pycharm_argv() -> List[str]:
    argv: List[str] = []
    cfg = PYCHARM_EVALUATE
    _append_optional_arg(argv, "--run-dir", cfg.get("run_dir"))
    _append_optional_arg(argv, "--checkpoint", cfg.get("checkpoint"))
    _append_optional_arg(argv, "--test-json", cfg.get("test_json"))
    _append_optional_arg(argv, "--output-dir", cfg.get("output_dir"))
    _append_optional_arg(argv, "--model-class", cfg.get("model_class"))
    _append_optional_arg(argv, "--score-floor", cfg.get("score_floor"))
    _append_optional_arg(argv, "--score-threshold", cfg.get("score_threshold"))
    _append_optional_arg(argv, "--class-score-thresholds", cfg.get("class_score_thresholds"))
    _append_optional_arg(argv, "--confmat-iou", cfg.get("confmat_iou"))
    _append_optional_arg(argv, "--curve-iou", cfg.get("curve_iou"))
    _append_optional_arg(argv, "--iou-min", cfg.get("iou_min"))
    _append_optional_arg(argv, "--iou-max", cfg.get("iou_max"))
    _append_optional_arg(argv, "--iou-step", cfg.get("iou_step"))
    _append_optional_arg(argv, "--threshold-points", cfg.get("threshold_points"))
    _append_optional_arg(argv, "--max-images", cfg.get("max_images"))
    _append_optional_arg(argv, "--num-overlays", cfg.get("num_overlays"))
    _append_optional_arg(argv, "--image-max-side", cfg.get("image_max_side"))
    _append_optional_arg(argv, "--seed", cfg.get("seed"))
    _append_optional_arg(argv, "--path-rewrite", cfg.get("path_rewrite"))
    _append_optional_arg(argv, "--images-root", cfg.get("images_root"))
    _append_optional_arg(argv, "--skip-missing-images", cfg.get("skip_missing_images"))
    _append_optional_arg(argv, "--no-plots", cfg.get("no_plots"))
    return argv


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
# Standard object detection evaluation
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
    class_score_thresholds: Dict[str, float]
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


def normalize_class_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def parse_class_score_thresholds(raw: Any) -> Dict[str, float]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        items = raw.items()
    else:
        text = str(raw).strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            items = payload.items()
        else:
            parts = [part.strip() for part in re.split(r"[;\n,]+", text) if part.strip()]
            parsed_items: List[Tuple[str, str]] = []
            for part in parts:
                if "=" in part:
                    name, value = part.rsplit("=", 1)
                elif ":" in part:
                    name, value = part.rsplit(":", 1)
                else:
                    raise ValueError(
                        "--class-score-thresholds entries must be NAME=VALUE pairs "
                        f"(got {part!r})"
                    )
                parsed_items.append((name.strip(), value.strip()))
            items = parsed_items

    thresholds: Dict[str, float] = {}
    for name, value in items:
        key = str(name).strip()
        if not key:
            continue
        threshold = float(value)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"Class score threshold for {key!r} must be between 0 and 1.")
        thresholds[key] = threshold
    return thresholds


def resolve_class_score_thresholds(
    labels: Sequence[str],
    configured: Dict[str, float],
    default_threshold: float,
) -> Dict[str, float]:
    resolved = {str(label): float(default_threshold) for label in labels}
    configured_by_key = {normalize_class_key(name): float(value) for name, value in configured.items()}

    for label in labels:
        label_text = str(label)
        label_key = normalize_class_key(label_text)
        threshold = configured.get(label_text)
        if threshold is None:
            threshold = configured_by_key.get(label_key)
        if threshold is None and any(token in label_key for token in ("leucocyte", "leukocyte", "wbc")):
            for alias in ("leu", "leucocyte", "leucocytes", "leukocyte", "leukocytes", "wbc"):
                if alias in configured_by_key:
                    threshold = configured_by_key[alias]
                    break
        if threshold is None and any(token in label_key for token in ("squamous", "epithelial")):
            for alias in ("epi", "epithelial", "epithelialcell", "epithelialcells", "squamous", "squamousepithelialcell"):
                if alias in configured_by_key:
                    threshold = configured_by_key[alias]
                    break
        if threshold is not None:
            resolved[label_text] = float(threshold)
    return resolved


def class_threshold_array(labels: Sequence[str], thresholds_by_label: Dict[str, float]) -> np.ndarray:
    return np.asarray(
        [float(thresholds_by_label.get(str(label), 0.0)) for label in labels],
        dtype=np.float32,
    )


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
    metadata_candidates = [
        (run_dir / "rfdetr_run" / "run_meta" / "model_architecture.json", ("model_name",)),
        (run_dir / "run_meta" / "model_architecture.json", ("model_name",)),
        (checkpoint.parent / "run_meta" / "model_architecture.json", ("model_name",)),
        (run_dir / "rfdetr_run" / "run_meta" / "train_kwargs.json", ("RFDETR_MODEL_CLS", "model_cls", "model_class")),
        (run_dir / "run_meta" / "train_kwargs.json", ("RFDETR_MODEL_CLS", "model_cls", "model_class")),
        (checkpoint.parent / "run_meta" / "train_kwargs.json", ("RFDETR_MODEL_CLS", "model_cls", "model_class")),
        (run_dir / "hpo_record.json", ("MODEL_CLS", "model_cls", "model_class")),
        (checkpoint.parent / "hpo_record.json", ("MODEL_CLS", "model_cls", "model_class")),
    ]

    for path, keys in metadata_candidates:
        if not path.exists():
            continue
        try:
            js = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in keys:
            name = str(js.get(key, "")).strip()
            if name in SUPPORTED_RFDETR_MODEL_NAME_SET:
                return name

    return infer_rfdetr_model_name_from_checkpoint_name(checkpoint.name.lower())


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _first_int_from_metadata(
    run_dir: Path,
    checkpoint: Path,
    keys: Sequence[str],
) -> Optional[int]:
    metadata_paths = [
        run_dir / "rfdetr_run" / "run_meta" / "train_kwargs.json",
        run_dir / "run_meta" / "train_kwargs.json",
        checkpoint.parent / "run_meta" / "train_kwargs.json",
        run_dir / "hpo_record.json",
        checkpoint.parent / "hpo_record.json",
    ]
    for path in metadata_paths:
        js = _read_json_if_exists(path)
        if js is None:
            continue
        for key in keys:
            raw = js.get(key)
            if raw is None or raw == "":
                continue
            try:
                return int(raw)
            except Exception:
                continue
    return None


def _int_from_mapping(mapping: Any, key: str) -> Optional[int]:
    if not isinstance(mapping, dict):
        return None
    raw = mapping.get(key)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _checkpoint_constructor_kwargs(checkpoint: Path) -> Dict[str, int]:
    if torch is None:
        return {}
    try:
        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    model_state = payload.get("model")
    if not isinstance(model_state, dict):
        return {}
    args_payload = payload.get("args")
    if not isinstance(args_payload, dict):
        args_payload = {}

    kwargs: Dict[str, int] = {}

    class_bias = model_state.get("class_embed.bias")
    if hasattr(class_bias, "shape") and len(class_bias.shape) >= 1:
        kwargs["num_classes"] = max(1, int(class_bias.shape[0]) - 1)

    group_detr = _int_from_mapping(args_payload, "group_detr")
    if group_detr is not None and group_detr > 0:
        kwargs["group_detr"] = group_detr

    num_select = _int_from_mapping(args_payload, "num_select")
    if num_select is not None and num_select > 0:
        kwargs["num_select"] = num_select

    refpoint_weight = model_state.get("refpoint_embed.weight")
    if hasattr(refpoint_weight, "shape") and len(refpoint_weight.shape) >= 1:
        total_query_slots = int(refpoint_weight.shape[0])
        if group_detr is not None and group_detr > 0 and total_query_slots % group_detr == 0:
            kwargs["num_queries"] = total_query_slots // group_detr

    patch_weight = model_state.get("backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight")
    patch_size: Optional[int] = None
    if hasattr(patch_weight, "shape") and len(patch_weight.shape) >= 4:
        patch_size = int(patch_weight.shape[-1])
        if patch_size > 0:
            kwargs["patch_size"] = patch_size

    pos_embed = model_state.get("backbone.0.encoder.encoder.embeddings.position_embeddings")
    if (
        patch_size is not None
        and patch_size > 0
        and hasattr(pos_embed, "shape")
        and len(pos_embed.shape) >= 2
    ):
        patch_positions = int(pos_embed.shape[1]) - 1
        grid_size = int(round(math.sqrt(max(0, patch_positions))))
        if grid_size > 0 and grid_size * grid_size == patch_positions:
            kwargs["positional_encoding_size"] = grid_size
            kwargs["resolution"] = grid_size * patch_size

    return kwargs


def load_model(model_class: str, checkpoint: Path, run_dir: Path):
    ensure_rfdetr_import()
    model_kwargs: Dict[str, Any] = {"pretrain_weights": str(checkpoint)}
    model_kwargs.update(_checkpoint_constructor_kwargs(checkpoint))

    resolution = _first_int_from_metadata(run_dir, checkpoint, ("resolution", "RESOLUTION"))
    if resolution is not None and "resolution" not in model_kwargs:
        model_kwargs["resolution"] = resolution

    model = instantiate_rfdetr_model(model_class, **model_kwargs)
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
            try:
                supports_threshold = "threshold" in inspect.signature(fn).parameters
            except (TypeError, ValueError):
                supports_threshold = False
            for inp in (img_pil, arr, ten):
                try:
                    if supports_threshold:
                        out = fn(inp, threshold=score_floor)
                    else:
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


def prediction_keep_mask(
    pred_scores: np.ndarray,
    pred_cls: np.ndarray,
    score_threshold: float,
    class_score_thresholds: Optional[np.ndarray] = None,
) -> np.ndarray:
    if pred_scores.size == 0:
        return np.zeros((0,), dtype=bool)
    if class_score_thresholds is None or class_score_thresholds.size == 0:
        return pred_scores >= float(score_threshold)

    thresholds = np.full((len(pred_scores),), float(score_threshold), dtype=np.float32)
    valid = (pred_cls >= 0) & (pred_cls < len(class_score_thresholds))
    if np.any(valid):
        thresholds[valid] = class_score_thresholds[pred_cls[valid]]
    return pred_scores >= thresholds


def confusion_matrix_with_background(
    samples: Sequence[Dict[str, Any]],
    n_classes: int,
    score_threshold: float,
    iou_thr: float,
    class_score_thresholds: Optional[np.ndarray] = None,
) -> np.ndarray:
    bg = n_classes
    cm = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
    for sample in samples:
        gt_boxes = sample["gt_boxes"]
        gt_cls = sample["gt_cls"]

        keep = prediction_keep_mask(sample["pred_scores"], sample["pred_cls"], score_threshold, class_score_thresholds)
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
    class_score_thresholds: Optional[np.ndarray] = None,
) -> Tuple[int, int, int]:
    """
    Class-aware matching.
    Returns (tp, fp, fn).
    """
    tp = fp = fn = 0
    for sample in samples:
        gt_boxes = sample["gt_boxes"]
        gt_cls = sample["gt_cls"]

        keep = prediction_keep_mask(sample["pred_scores"], sample["pred_cls"], score_threshold, class_score_thresholds)
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


def count_error_metrics(
    samples: Sequence[Dict[str, Any]],
    class_names: Sequence[str],
    score_threshold: float,
    class_score_thresholds: Optional[np.ndarray] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Summarize raw predicted-versus-annotated counts per image and class."""
    per_image: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    for class_idx, class_name in enumerate(class_names):
        signed_errors: List[int] = []
        for sample in samples:
            keep = prediction_keep_mask(
                sample["pred_scores"], sample["pred_cls"], score_threshold, class_score_thresholds
            )
            annotated = int(np.sum(sample["gt_cls"] == class_idx))
            predicted = int(np.sum(sample["pred_cls"][keep] == class_idx))
            signed = predicted - annotated
            signed_errors.append(signed)
            per_image.append(
                {
                    "image_id": int(sample["img_id"]),
                    "class": class_name,
                    "annotated_count": annotated,
                    "predicted_count": predicted,
                    "signed_error": signed,
                    "absolute_error": abs(signed),
                }
            )
        errors = np.asarray(signed_errors, dtype=np.float64)
        summary.append(
            {
                "class": class_name,
                "n_images": int(len(errors)),
                "mean_absolute_error": float(np.mean(np.abs(errors))),
                "median_absolute_error": float(np.median(np.abs(errors))),
                "mean_signed_error": float(np.mean(errors)),
            }
        )
    return per_image, summary


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

    def class_color(name: str) -> Tuple[int, int, int]:
        key = str(name).strip().lower()
        if "leucocyte" in key or "leukocyte" in key:
            return (220, 30, 30)
        if "epithelial" in key or "squamous" in key:
            return (30, 95, 220)
        return (255, 160, 0)

    for box, name, score in zip(pred_boxes, pred_names, pred_scores):
        x1, y1, x2, y2 = sc(box)
        color = class_color(name)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, y1 + 2), f"{float(score):.2f} {name}", fill=color)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def matched_correct_predictions(
    sample: Dict[str, Any],
    score_threshold: float,
    iou_thr: float,
    class_score_thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep = prediction_keep_mask(sample["pred_scores"], sample["pred_cls"], score_threshold, class_score_thresholds)
    pred_boxes = sample["pred_boxes"][keep]
    pred_scores = sample["pred_scores"][keep]
    pred_cls = sample["pred_cls"][keep]

    if len(sample["gt_boxes"]) == 0 or len(pred_boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    ious = iou_matrix(sample["gt_boxes"], pred_boxes)
    valid = sample["gt_cls"][:, None] == pred_cls[None, :]
    matches = greedy_match(ious, iou_thr=iou_thr, valid_pairs=valid)
    pred_indices = [pj for _, pj in matches]
    if not pred_indices:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return pred_boxes[pred_indices], pred_scores[pred_indices], pred_cls[pred_indices]


def save_confusion_csv(path: Path, labels: List[str], cm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + labels)
        for idx, row in enumerate(cm):
            writer.writerow([labels[idx]] + row.tolist())


def save_confusion_figure(
    path: Path,
    labels: List[str],
    cm: np.ndarray,
    title: str = "Object Detection Confusion Matrix",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[EVALUATE][WARN] matplotlib not available; skipping confusion matrix PNG.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    row_totals = cm.sum(axis=1, keepdims=True)
    pct = np.divide(
        cm,
        row_totals,
        out=np.zeros_like(cm, dtype=np.float64),
        where=row_totals != 0,
    ) * 100.0

    fig_size = max(7.0, 1.85 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(pct, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=100.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized percentage")

    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("Predicted class", fontsize=12)
    ax.set_ylabel("Ground truth class", fontsize=12)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels, rotation=30, va="center")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if pct[i, j] >= 50.0 else "black"
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}\n{pct[i, j]:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
            )

    ax.set_ylim(len(labels) - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_iou_ap_figure(
    path: Path,
    iou_rows: Sequence[Dict[str, Any]],
    per_class_iou_rows: Sequence[Dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[EVALUATE][WARN] matplotlib not available; skipping mAP-by-IoU PNG.")
        return

    if not iou_rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    x_vals = [float(row["iou_threshold"]) for row in iou_rows]
    y_vals = [float(row["AP_all"]) for row in iou_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_vals, y_vals, marker="o", linewidth=2.3, label="mAP")

    class_rows: Dict[str, List[Tuple[float, float]]] = {}
    for row in per_class_iou_rows:
        class_rows.setdefault(str(row["class"]), []).append((float(row["iou_threshold"]), float(row["AP"])))
    for class_name, points in sorted(class_rows.items()):
        points.sort(key=lambda item: item[0])
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker=".",
            linewidth=1.6,
            alpha=0.85,
            label=f"AP {class_name}",
        )

    ax.set_title("AP by IoU Threshold")
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("AP / mAP")
    ax.set_xlim(min(x_vals), max(x_vals))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


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


def timestamped_output_dir(base_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = base_dir.parent / f"{base_dir.name}_{stamp}"
    suffix = 2
    while output_dir.exists():
        output_dir = base_dir.parent / f"{base_dir.name}_{stamp}_{suffix:02d}"
        suffix += 1
    return output_dir


def build_eval_config(args: argparse.Namespace) -> EvalConfig:
    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint or (run_dir / "rfdetr_run" / "checkpoint_best_total.pth")
    checkpoint = checkpoint.resolve()
    test_json = args.test_json or (run_dir / "test" / "_annotations.coco.json")
    test_json = test_json.resolve()
    output_dir = args.output_dir or (run_dir / "rfdetr_run" / "eval_custom")
    output_dir = timestamped_output_dir(output_dir.resolve())

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
    class_score_thresholds = parse_class_score_thresholds(args.class_score_thresholds)

    return EvalConfig(
        run_dir=run_dir,
        checkpoint=checkpoint,
        test_json=test_json,
        output_dir=output_dir,
        model_class=model_class,
        score_floor=float(args.score_floor),
        score_threshold=float(args.score_threshold),
        class_score_thresholds=class_score_thresholds,
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
    resolved_class_thresholds = resolve_class_score_thresholds(
        labels,
        cfg.class_score_thresholds,
        cfg.score_threshold,
    )
    class_thresholds = class_threshold_array(labels, resolved_class_thresholds)
    print("[EVALUATE] score floor   :", f"{cfg.score_floor:.4f}")
    print("[EVALUATE] fallback thr  :", f"{cfg.score_threshold:.4f}")
    print("[EVALUATE] class thresh  :", resolved_class_thresholds)

    model = load_model(cfg.model_class, cfg.checkpoint, cfg.run_dir)

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
    save_iou_ap_figure(
        cfg.output_dir / "map_by_iou_threshold.png",
        iou_rows=iou_rows,
        per_class_iou_rows=per_class_iou_rows,
    )

    cm = confusion_matrix_with_background(
        samples=samples,
        n_classes=n_classes,
        score_threshold=cfg.score_threshold,
        iou_thr=cfg.confmat_iou,
        class_score_thresholds=class_thresholds,
    )
    cm_labels = labels + ["background"]
    save_confusion_csv(cfg.output_dir / "confusion_matrix.csv", cm_labels, cm)
    save_confusion_figure(
        cfg.output_dir / "confusion_matrix.png",
        cm_labels,
        cm,
        title="Object Detection Confusion Matrix",
    )
    json_dump(
        cfg.output_dir / "confusion_matrix.json",
        {
            "labels": cm_labels,
            "matrix": cm.tolist(),
            "prediction_score_floor": cfg.score_floor,
            "score_threshold": cfg.score_threshold,
            "class_score_thresholds": resolved_class_thresholds,
            "iou_threshold": cfg.confmat_iou,
            "matching": "class_agnostic_iou_matched_detection",
            "background": True,
            "note": "Rows are ground truth, columns are predictions. Class-specific score thresholds are applied when present. Wrong-class IoU matches are counted off diagonal. Background column means missed GT; background row means unmatched predictions.",
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

    operating_tp, operating_fp, operating_fn = detection_counts(
        samples,
        score_threshold=cfg.score_threshold,
        iou_thr=cfg.curve_iou,
        class_score_thresholds=class_thresholds,
    )
    operating_precision = float(operating_tp / (operating_tp + operating_fp)) if (operating_tp + operating_fp) > 0 else 0.0
    operating_recall = float(operating_tp / (operating_tp + operating_fn)) if (operating_tp + operating_fn) > 0 else 0.0
    operating_f1 = (
        float((2 * operating_precision * operating_recall) / (operating_precision + operating_recall))
        if (operating_precision + operating_recall) > 0
        else 0.0
    )
    operating_point = {
        "score_threshold": cfg.score_threshold,
        "class_score_thresholds": resolved_class_thresholds,
        "tp": int(operating_tp),
        "fp": int(operating_fp),
        "fn": int(operating_fn),
        "precision": operating_precision,
        "recall": operating_recall,
        "f1": operating_f1,
        "fp_per_image": float(operating_fp / max(1, len(samples))),
    }

    per_image_count_rows, count_error_rows = count_error_metrics(
        samples,
        labels,
        score_threshold=cfg.score_threshold,
        class_score_thresholds=class_thresholds,
    )
    write_csv(
        cfg.output_dir / "per_image_count_errors.csv",
        ["image_id", "class", "annotated_count", "predicted_count", "signed_error", "absolute_error"],
        per_image_count_rows,
    )
    write_csv(
        cfg.output_dir / "count_error_metrics.csv",
        ["class", "n_images", "mean_absolute_error", "median_absolute_error", "mean_signed_error"],
        count_error_rows,
    )

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
        eligible = [
            s
            for s in samples
            if len(matched_correct_predictions(s, cfg.score_threshold, cfg.confmat_iou, class_thresholds)[0]) > 0
        ]
        random.shuffle(eligible)
        chosen = eligible[: cfg.num_overlays]
        for idx, sample in enumerate(chosen, start=1):
            pred_boxes, pred_scores, pred_cls = matched_correct_predictions(
                sample,
                cfg.score_threshold,
                cfg.confmat_iou,
                class_thresholds,
            )

            pred_names = [labels[int(c)] for c in pred_cls]
            draw_overlay(
                img_path=Path(sample["img_path"]),
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
        "class_score_thresholds": resolved_class_thresholds,
        "confmat_iou": cfg.confmat_iou,
        "curve_iou": cfg.curve_iou,
        "evaluation_policy": {
            "ranking_metrics": (
                "COCO AP/AR use all predictions retained at the "
                f"{cfg.score_floor:g} numerical score floor and do not use the "
                "calibrated operating thresholds."
            ),
            "threshold_dependent_metrics": (
                "The confusion matrix and operating-point precision/recall/F1 "
                "use the independently calibrated class thresholds."
            ),
            "threshold_sweep": (
                "Saved for descriptive sensitivity analysis only; no threshold "
                "is selected or optimized on this held-out test set."
            ),
        },
        "iou_sweep": {
            "min": cfg.iou_min,
            "max": cfg.iou_max,
            "step": cfg.iou_step,
            "rows": iou_rows,
        },
        "coco_standard": coco_std,
        "operating_point": operating_point,
        "count_error_metrics": count_error_rows,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "roc_note": roc_note,
        "outputs": {
            "predictions": str(predictions_path),
            "iou_sweep_metrics": str(cfg.output_dir / "iou_sweep_metrics.csv"),
            "per_class_metrics": str(cfg.output_dir / "per_class_metrics.csv"),
            "count_error_metrics": str(cfg.output_dir / "count_error_metrics.csv"),
            "per_image_count_errors": str(cfg.output_dir / "per_image_count_errors.csv"),
            "confusion_matrix_csv": str(cfg.output_dir / "confusion_matrix.csv"),
            "confusion_matrix_json": str(cfg.output_dir / "confusion_matrix.json"),
            "confusion_matrix_figure": str(cfg.output_dir / "confusion_matrix.png"),
            "map_by_iou_figure": str(cfg.output_dir / "map_by_iou_threshold.png"),
            "threshold_metrics": str(cfg.output_dir / "threshold_metrics.csv"),
            "pr_curve": str(cfg.output_dir / "pr_curve_overall.csv"),
            "roc_curve": str(cfg.output_dir / "roc_curve_overall.csv"),
        },
    }
    json_dump(cfg.output_dir / "eval_summary.json", summary)

    print("\n[EVALUATE] Done")
    print(f"[EVALUATE] AP@50:95={coco_std['AP@50:95']:.4f}  AP@50={coco_std['AP@50']:.4f}  AP@75={coco_std['AP@75']:.4f}")
    print(
        "[EVALUATE] Operating F1="
        f"{operating_f1:.4f} "
        f"(precision={operating_precision:.4f}, recall={operating_recall:.4f}, "
        f"fp/image={operating_point['fp_per_image']:.2f})"
    )
    print(
        "[EVALUATE] Threshold sweep is descriptive only; "
        "no threshold was selected on the held-out test set."
    )
    print(f"[EVALUATE] Summary JSON -> {(cfg.output_dir / 'eval_summary.json').resolve()}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standard RF-DETR object detection evaluation on a COCO test split."
    )
    run_dir_env = os.getenv("EVAL_RUN_DIR", "").strip()
    ckpt_env = os.getenv("EVAL_CHECKPOINT", "").strip()
    test_json_env = os.getenv("EVAL_TEST_JSON", "").strip()
    output_env = os.getenv("EVAL_OUTPUT_DIR", "").strip()
    images_root_env = os.getenv("EVAL_IMAGES_ROOT", "").strip()

    parser.add_argument("--run-dir", type=Path, default=Path(run_dir_env) if run_dir_env else Path("."))
    parser.add_argument("--checkpoint", type=Path, default=Path(ckpt_env) if ckpt_env else None)
    parser.add_argument("--test-json", type=Path, default=Path(test_json_env) if test_json_env else None)
    parser.add_argument("--output-dir", type=Path, default=Path(output_env) if output_env else None)
    parser.add_argument(
        "--model-class",
        type=str,
        default=os.getenv("EVAL_MODEL_CLASS", "auto"),
        choices=supported_rfdetr_model_names(include_auto=True),
    )
    parser.add_argument("--score-floor", type=float, default=float(os.getenv("EVAL_SCORE_FLOOR", "0.001")))
    parser.add_argument("--score-threshold", type=float, default=float(os.getenv("EVAL_SCORE_THRESHOLD", "0.001")))
    parser.add_argument(
        "--class-score-thresholds",
        type=str,
        default=os.getenv("EVAL_CLASS_SCORE_THRESHOLDS", DEFAULT_CLASS_SCORE_THRESHOLDS_TEXT),
        help="Per-class acceptance thresholds, e.g. 'Leucocyte=0.30;Squamous Epithelial Cell=0.65'.",
    )
    parser.add_argument("--confmat-iou", type=float, default=float(os.getenv("EVAL_CONFMAT_IOU", "0.50")))
    parser.add_argument("--curve-iou", type=float, default=float(os.getenv("EVAL_CURVE_IOU", "0.50")))
    parser.add_argument("--iou-min", type=float, default=float(os.getenv("EVAL_IOU_MIN", "0.10")))
    parser.add_argument("--iou-max", type=float, default=float(os.getenv("EVAL_IOU_MAX", "0.95")))
    parser.add_argument("--iou-step", type=float, default=float(os.getenv("EVAL_IOU_STEP", "0.05")))
    parser.add_argument("--threshold-points", type=int, default=int(os.getenv("EVAL_THRESHOLD_POINTS", "51")))
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--num-overlays", type=int, default=int(os.getenv("EVAL_NUM_OVERLAYS", "8")))
    parser.add_argument("--image-max-side", type=int, default=int(os.getenv("EVAL_IMAGE_MAX_SIDE", "1600")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("EVAL_SEED", "42")))
    parser.add_argument(
        "--path-rewrite",
        type=str,
        default=os.getenv("EVAL_PATH_REWRITE", ""),
        help="Path rewrite rules in FROM=TO;FROM=TO format.",
    )
    parser.add_argument("--images-root", type=Path, default=Path(images_root_env) if images_root_env else None)
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        default=env_bool("EVAL_SKIP_MISSING_IMAGES", False),
    )
    parser.add_argument("--no-plots", action="store_true", default=env_bool("EVAL_NO_PLOTS", False))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    if argv is None and PYCHARM_USE_TOP_LEVEL_CONFIG and not sys.argv[1:]:
        argv = build_pycharm_argv()
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    run_evaluate_mode(args)


if __name__ == "__main__":
    main()
