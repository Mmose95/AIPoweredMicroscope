#!/usr/bin/env python
"""RF-DETR inference helpers for the QA workflow UI.

This module is the single processing dependency for ``qa_workflow_ui.py``.
It owns checkpoint inspection, RF-DETR loading, SAHI sliced inference,
duplicate suppression, object counting, and conversion to downstream QA labels.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import math
import re
import sys
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for import_path in (PROJECT_ROOT, SCRIPT_DIR):
    import_str = str(import_path)
    if import_str not in sys.path:
        sys.path.insert(0, import_str)

from rfdetr_model_registry import (  # noqa: E402
    SUPPORTED_RFDETR_MODEL_NAME_SET,
    infer_rfdetr_model_name_from_checkpoint_name,
    instantiate_rfdetr_model,
)


ACTIVE_PRESET = "two_class"
DEFAULT_IMAGES_ROOT = r"E:\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment_Cleaned"
DEFAULT_CLASS_NAMES = [
    "Leucocyte",
    "Squamous Epithelial Cell",
]
CLASS_SCORE_THRESHOLDS: Dict[str, float] = {
    "Leucocyte": 0.36,
    "Squamous Epithelial Cell": 0.35,
}
DOWNSTREAM_LABELS = {
    1: "Qualified",
    2: "Partially Qualified",
    3: "Not Qualified",
}
DOWNSTREAM_PRESETS = {
    "two_class": {
        "target_name": "Downstream",
        "checkpoint": (
            r"E:\PHD\Results\Quality Assessment\FINAL_B200"
            r"\session_20260618_113853\TwoClass\HPO_Config_009\checkpoint_best_ema.pth"
        ),
        "images_root": DEFAULT_IMAGES_ROOT,
        "model_class": "auto",
        "class_names": DEFAULT_CLASS_NAMES,
        "class_score_thresholds": CLASS_SCORE_THRESHOLDS,
    },
}
DEFAULT_CHECKPOINT = Path(
    DOWNSTREAM_PRESETS[ACTIVE_PRESET]["checkpoint"]
)

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
SAMPLE_RE = re.compile(r"(Sample\s*\d+|Sample\d+)", re.IGNORECASE)
COORD_RE = re.compile(r"BF\.(\d+)_(\d+)", re.IGNORECASE)
SEVERITY = {
    "Qualified": 0,
    "Partially Qualified": 1,
    "Not Qualified": 2,
}

DEFAULT_SCORE_FLOOR = 0.001
DEFAULT_SCORE_THRESHOLD = 0.30
DEFAULT_SLICE_HEIGHT = 640
DEFAULT_SLICE_WIDTH = 640
DEFAULT_OVERLAP_HEIGHT_RATIO = 0.20
DEFAULT_OVERLAP_WIDTH_RATIO = 0.20
DEFAULT_PERFORM_STANDARD_PRED = False
DEFAULT_POSTPROCESS_TYPE = "GREEDYNMM"
DEFAULT_POSTPROCESS_MATCH_METRIC = "IOU"
DEFAULT_POSTPROCESS_MATCH_THRESHOLD = 0.50
DEFAULT_POSTPROCESS_CLASS_AGNOSTIC = False

CROSS_CLASS_DUPLICATE_IOS_THRESHOLD = 0.75
CROSS_CLASS_DUPLICATE_IOU_THRESHOLD = 0.35
CROSS_CLASS_DUPLICATE_AREA_RATIO_THRESHOLD = 0.55


@dataclass
class FOVInferenceResult:
    predicted_label_id: int
    predicted_label: str
    n_leucocyte: int
    n_squamous_epithelial_cell: int
    leucocyte_score: int
    squamous_epithelial_score: int
    total_quality_score: int
    n_predictions_raw: int
    n_predictions_kept_before_duplicate_suppression: int
    n_cross_class_duplicates_suppressed: int
    n_predictions_kept: int
    pred_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    pred_scores: list[float] = field(default_factory=list)
    pred_cls: list[int] = field(default_factory=list)


@dataclass
class FOVRecord:
    ingest_index: int
    coord_x: int
    coord_y: int
    image_path: Path
    image_name: str
    image_width: int = 1
    image_height: int = 1
    stage: str = "Pending"
    result: FOVInferenceResult | None = None
    error: str = ""
    result_version: int = 0
    inference_seconds: float | None = None


@dataclass
class SampleSession:
    sample_id: str
    sample_dir: Path
    checkpoint_path: Path
    model_class: str
    model_resolution: int | None
    class_names: list[str]
    class_score_thresholds: dict[str, float]
    fovs: list[FOVRecord]

    @property
    def grid_width(self) -> int:
        return max((fov.coord_x for fov in self.fovs), default=0) + 1

    @property
    def grid_height(self) -> int:
        return max((fov.coord_y for fov in self.fovs), default=0) + 1

    @property
    def by_position(self) -> dict[tuple[int, int], FOVRecord]:
        return {(fov.coord_x, fov.coord_y): fov for fov in self.fovs}


@dataclass
class InferenceRuntime:
    get_sliced_prediction: Any
    sahi_model: Any
    class_names: list[str]
    class_score_thresholds: dict[str, float]
    model_class: str
    model_resolution: int | None


def check_dependencies() -> None:
    missing: list[str] = []
    if np is None:
        missing.append("numpy")
    if Image is None or ImageDraw is None or ImageFont is None:
        missing.append("Pillow")
    if missing:
        raise ImportError("Missing dependencies: " + ", ".join(missing))


def ensure_inference_deps() -> None:
    check_dependencies()
    if torch is None:
        raise ImportError("Missing dependencies: torch")
    import_sahi()


def configure_torch_inference_backend() -> None:
    if torch is None:
        return
    try:
        torch.set_grad_enabled(False)
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


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
            resolved_features = [stage_names[idx] for idx in resolved_indices]
            return resolved_features, resolved_indices

        if out_indices is None:
            feature_to_index = {name: idx for idx, name in enumerate(stage_names)}
            resolved_features = [str(value) for value in out_features]
            resolved_indices = [feature_to_index[name] for name in resolved_features]
            return resolved_features, resolved_indices

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


def force_repo_local_rfdetr() -> None:
    local_pkg_dir = PROJECT_ROOT / "rfdetr_local"
    init_py = local_pkg_dir / "__init__.py"
    if not init_py.exists():
        raise FileNotFoundError(f"Repo-local RF-DETR package was not found at {local_pkg_dir}")

    project_root_str = str(PROJECT_ROOT)
    local_pkg_dir_str = str(local_pkg_dir)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    if local_pkg_dir_str not in sys.path:
        sys.path.insert(0, local_pkg_dir_str)

    patch_transformers_torch_compat()
    _patch_transformers_pruning_compat()
    _patch_transformers_backbone_compat()
    for key in list(sys.modules):
        if key == "rfdetr" or key.startswith("rfdetr."):
            sys.modules.pop(key, None)

    spec = importlib.util.spec_from_file_location(
        "rfdetr",
        init_py,
        submodule_search_locations=[str(local_pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {init_py}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["rfdetr"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ImportError(
            f"Repo-local RF-DETR import failed from {local_pkg_dir}: {exc}."
            f"{build_env_mismatch_hint(exc)}"
        ) from exc


def import_sahi() -> tuple[Any, Any, Any]:
    try:
        from sahi.models.base import DetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.prediction import ObjectPrediction
    except Exception as exc:
        raise ImportError(
            "SAHI is required for full-FOV sliced inference. Install it in the active environment, e.g. "
            "`python -m pip install sahi`."
        ) from exc
    return DetectionModel, get_sliced_prediction, ObjectPrediction


def infer_checkpoint_runtime_metadata(checkpoint: Path) -> tuple[Optional[int], Optional[list[str]]]:
    if torch is None:
        return None, None
    try:
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except Exception:
        return None, None

    args = ckpt.get("args") if isinstance(ckpt, dict) else None
    if args is None:
        return None, None

    num_classes = None
    class_names = None
    try:
        num_classes = int(getattr(args, "num_classes", None))
    except Exception:
        try:
            num_classes = int(args.get("num_classes")) if isinstance(args, dict) and args.get("num_classes") else None
        except Exception:
            num_classes = None
    try:
        raw_names = getattr(args, "class_names", None)
        if raw_names is None and isinstance(args, dict):
            raw_names = args.get("class_names")
        if raw_names:
            class_names = [str(name).strip() for name in raw_names if str(name).strip()]
    except Exception:
        class_names = None
    return num_classes, class_names


def infer_model_class(model_meta_root: Path, checkpoint: Path) -> str:
    metadata_candidates = [
        (model_meta_root / "rfdetr_run" / "run_meta" / "model_architecture.json", ("model_name",)),
        (model_meta_root / "run_meta" / "model_architecture.json", ("model_name",)),
        (checkpoint.parent / "run_meta" / "model_architecture.json", ("model_name",)),
        (model_meta_root / "rfdetr_run" / "run_meta" / "train_kwargs.json", ("RFDETR_MODEL_CLS", "model_cls", "model_class")),
        (model_meta_root / "run_meta" / "train_kwargs.json", ("RFDETR_MODEL_CLS", "model_cls", "model_class")),
        (checkpoint.parent / "run_meta" / "train_kwargs.json", ("RFDETR_MODEL_CLS", "model_cls", "model_class")),
        (model_meta_root / "hpo_record.json", ("MODEL_CLS", "model_cls", "model_class")),
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


def _read_json_if_exists(path: Path) -> Optional[dict[str, Any]]:
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


def infer_model_resolution(checkpoint: Path) -> Optional[int]:
    resolution = _first_int_from_metadata(checkpoint.parent, checkpoint, ("resolution", "RESOLUTION"))
    if resolution is not None:
        return resolution
    if torch is None:
        return None
    try:
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        args = ckpt.get("args", None) if isinstance(ckpt, dict) else None
        if args is None:
            return None
        if isinstance(args, dict):
            val = args.get("resolution")
            return int(val) if val is not None else None
        val = getattr(args, "resolution", None)
        return int(val) if val is not None else None
    except Exception:
        return None


def _checkpoint_constructor_kwargs(checkpoint: Path) -> dict[str, int]:
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

    kwargs: dict[str, int] = {}
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


def inspect_checkpoint(checkpoint_path: Path) -> tuple[str, int | None, list[str], dict[str, float]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    model_class = infer_model_class(checkpoint_path.parent, checkpoint_path)
    model_resolution = infer_model_resolution(checkpoint_path)
    ckpt_num_classes, ckpt_class_names = infer_checkpoint_runtime_metadata(checkpoint_path)

    class_names = list(ckpt_class_names or DEFAULT_CLASS_NAMES)
    if ckpt_num_classes is not None and ckpt_num_classes > len(class_names):
        class_names = list(class_names) + [f"Class {idx}" for idx in range(len(class_names), ckpt_num_classes)]

    class_score_thresholds = {
        name: float(CLASS_SCORE_THRESHOLDS.get(name, DEFAULT_SCORE_THRESHOLD))
        for name in class_names
    }
    return model_class, model_resolution, class_names, class_score_thresholds


def load_model_for_fov(
    model_class: str,
    checkpoint: Path,
    resolution: Optional[int],
    num_classes: Optional[int],
    class_names: Optional[Sequence[str]],
) -> Any:
    force_repo_local_rfdetr()
    model_kwargs: dict[str, Any] = {"pretrain_weights": str(checkpoint)}
    model_kwargs.update(_checkpoint_constructor_kwargs(checkpoint))
    if resolution is not None:
        model_kwargs["resolution"] = int(resolution)
    if num_classes is not None:
        model_kwargs["num_classes"] = int(num_classes)

    model = instantiate_rfdetr_model(model_class, **model_kwargs)
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
    for candidate in (model, getattr(model, "model", None), getattr(model, "model_ema", None)):
        if hasattr(candidate, "eval"):
            try:
                candidate.eval()
            except Exception:
                pass
    return model


def predict_one_image(model: Any, img_pil: Any, score_floor: float) -> tuple[Any, Any, Any]:
    if np is None or torch is None:
        raise ImportError("numpy and torch are required for RF-DETR prediction.")

    # RF-DETR accepts PIL directly. Avoid building an unused float tensor for
    # every SAHI tile; retain lazy array/tensor fallbacks for other wrappers.
    arr = ten = None

    def inputs():
        nonlocal arr, ten
        yield img_pil
        if arr is None:
            arr = np.array(img_pil.convert("RGB"))
        yield arr
        if ten is None:
            ten = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        yield ten

    out = None
    for name in ("predict", "infer", "inference", "forward_inference"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                supports_threshold = "threshold" in inspect.signature(fn).parameters
            except (TypeError, ValueError):
                supports_threshold = False
            for inp in inputs():
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
            if ten is None:
                for _ in inputs():
                    pass
            out = forward(ten)
        else:
            raise RuntimeError("Model has no predict/infer/forward_inference/forward method.")

    try:
        import supervision as sv

        if isinstance(out, sv.Detections):
            boxes = out.xyxy.astype(np.float32)
            scores = (
                np.ones((len(boxes),), dtype=np.float32)
                if getattr(out, "confidence", None) is None
                else out.confidence.astype(np.float32)
            )
            labels = (
                np.zeros((len(boxes),), dtype=np.int64)
                if getattr(out, "class_id", None) is None
                else out.class_id.astype(np.int64)
            )
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

    def to_np(x: Any) -> Any:
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    boxes_np = to_np(boxes).astype(np.float32)
    scores_np = to_np(scores).astype(np.float32) if scores is not None else np.ones((len(boxes_np),), dtype=np.float32)
    labels_np = to_np(labels).astype(np.int64) if labels is not None else np.zeros((len(boxes_np),), dtype=np.int64)

    keep = scores_np >= score_floor
    return boxes_np[keep], scores_np[keep], labels_np[keep]


def parse_sample_id(raw_value: str) -> str:
    match = SAMPLE_RE.search(raw_value)
    if match:
        return match.group(1).replace(" ", "")
    return Path(raw_value).stem.replace(" ", "")


def explicit_sample_id(raw_value: str) -> str | None:
    match = SAMPLE_RE.search(raw_value)
    if not match:
        return None
    return match.group(1).replace(" ", "")


def parse_coordinates(raw_value: str) -> tuple[int, int]:
    match = COORD_RE.search(raw_value)
    if match:
        return int(match.group(1)), int(match.group(2))
    fallback = re.search(r"(\d+)_(\d+)(?=\.[^.]+$)", raw_value)
    if fallback:
        return int(fallback.group(1)), int(fallback.group(2))
    raise ValueError(f"Could not parse coordinates from '{raw_value}'")


def discover_images(sample_dirs: Sequence[Path]) -> list[Path]:
    images: list[Path] = []
    for sample_dir in sample_dirs:
        direct_files = [
            path
            for path in sample_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if direct_files:
            images.extend(sorted(direct_files, key=lambda p: str(p).lower()))
            continue
        for path in sample_dir.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if "patch" in str(path).lower():
                continue
            images.append(path)
    return sorted(images, key=lambda p: str(p).lower())


def discover_sample_images(sample_dir: Path) -> list[Path]:
    if not sample_dir.exists() or not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample folder does not exist: {sample_dir}")

    direct_images = [
        path
        for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and "patch" not in str(path).lower()
    ]
    if direct_images:
        return sorted(direct_images, key=lambda path: str(path).lower())

    child_sample_dirs = [
        path
        for path in sample_dir.iterdir()
        if path.is_dir() and explicit_sample_id(path.name) is not None
    ]
    if child_sample_dirs:
        raise RuntimeError(
            f"{sample_dir} looks like a parent folder containing sample folders. "
            "Open Add, browse to this parent folder, select one or more Sample folders in the list, then click Add selected."
        )

    images = [
        path
        for path in discover_images([sample_dir])
        if path.suffix.lower() in IMAGE_EXTENSIONS and "patch" not in str(path).lower()
    ]

    if not images:
        raise RuntimeError(f"No microscope images were found under {sample_dir}")
    return images


def _read_image_size(image_path: Path) -> tuple[int, int]:
    try:
        with Image.open(image_path) as raw_img:
            return max(1, int(raw_img.width)), max(1, int(raw_img.height))
    except Exception:
        return 1, 1


def build_sample_session(
    checkpoint_path: Path, sample_dir: Path,
    checkpoint_metadata: tuple[str, int | None, list[str], dict[str, float]] | None = None,
) -> SampleSession:
    model_class, model_resolution, class_names, class_score_thresholds = (
        checkpoint_metadata if checkpoint_metadata is not None else inspect_checkpoint(checkpoint_path)
    )
    image_paths = discover_sample_images(sample_dir)

    parsed_rows: list[tuple[int, int, Path, str]] = []
    explicit_sample_ids: set[str] = set()
    for path in image_paths:
        image_name = path.name
        coord_x, coord_y = parse_coordinates(image_name)
        image_sample_id = explicit_sample_id(image_name)
        if image_sample_id is not None:
            explicit_sample_ids.add(image_sample_id)
        parsed_rows.append((coord_x, coord_y, path, image_name))

    folder_sample_id = explicit_sample_id(sample_dir.name) or sample_dir.name.replace(" ", "")
    if not explicit_sample_ids:
        sample_id = folder_sample_id
    elif len(explicit_sample_ids) == 1:
        sample_id = next(iter(explicit_sample_ids))
    else:
        raise RuntimeError(
            "Selected folder contains images from multiple samples. "
            f"Detected sample ids: {', '.join(sorted(explicit_sample_ids))}."
        )

    parsed_rows.sort(key=lambda item: (item[1], item[0], item[3].lower()))
    fovs = [
        FOVRecord(
            ingest_index=index,
            coord_x=coord_x,
            coord_y=coord_y,
            image_path=image_path,
            image_name=image_name,
            image_width=image_size[0],
            image_height=image_size[1],
        )
        for index, (coord_x, coord_y, image_path, image_name) in enumerate(parsed_rows)
        for image_size in [_read_image_size(image_path)]
    ]

    return SampleSession(
        sample_id=sample_id,
        sample_dir=sample_dir,
        checkpoint_path=checkpoint_path,
        model_class=model_class,
        model_resolution=model_resolution,
        class_names=list(class_names),
        class_score_thresholds=dict(class_score_thresholds),
        fovs=fovs,
    )


def threshold_for_class_idx(
    class_idx: int,
    class_names: Sequence[str],
    class_score_thresholds: dict[str, float],
    default_threshold: float,
) -> float:
    if 0 <= int(class_idx) < len(class_names):
        return float(class_score_thresholds.get(class_names[int(class_idx)], default_threshold))
    return float(default_threshold)


def per_class_keep_mask(
    pred_scores: Any,
    pred_cls: Any,
    class_names: Sequence[str],
    class_score_thresholds: dict[str, float],
    default_threshold: float,
) -> Any:
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


def normalize_model_class_ids(pred_labels: Any, class_names: Sequence[str]) -> Any:
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
    boxes: Any,
    image_w: int,
    image_h: int,
    min_size: float = 1e-3,
) -> Any:
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


def _measure_text(draw: Any, text: str, font: Any) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return int(right - left), int(bottom - top)
    return draw.textsize(text, font=font)


def draw_text_with_outline(draw: Any, xy: tuple[int, int], text: str, fill: tuple[int, int, int], font: Any) -> None:
    x, y = xy
    outline = (0, 0, 0)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, fill=outline, font=font)
    draw.text((x, y), text, fill=fill, font=font)


def _score_anchor(box: Sequence[float], image_w: int, image_h: int, text_w: int, text_h: int) -> tuple[int, int]:
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


def class_color_map(class_names: Sequence[str]) -> dict[int, tuple[int, int, int]]:
    palette = [
        (220, 30, 30),
        (20, 170, 80),
        (255, 140, 0),
        (40, 120, 220),
        (180, 80, 200),
        (0, 180, 180),
    ]
    return {i: palette[i % len(palette)] for i in range(len(class_names))}


def build_sahi_model(model: Any, class_names: Sequence[str], confidence_threshold: float) -> Any:
    DetectionModel, _, ObjectPrediction = import_sahi()

    class RFDETRLocalSahiModel(DetectionModel):  # type: ignore[misc]
        required_packages: list[str] = []

        def set_model(self, model: Any, **kwargs: Any) -> None:
            self.model = model

        def load_model(self) -> None:
            raise RuntimeError("Use an already loaded RF-DETR model instance with this wrapper.")

        def perform_inference(self, image: Any) -> None:
            img_pil = Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
            boxes, scores, labels = predict_one_image(
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
            shift_amount_list: Optional[list[list[int]]] = None,
            full_shape_list: Optional[list[list[int]]] = None,
        ) -> None:
            try:
                from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
            except Exception as exc:
                raise ImportError("SAHI compatibility helpers are unavailable.") from exc

            shift_amount_list = fix_shift_amount_list(shift_amount_list)
            full_shape_list = fix_full_shape_list(full_shape_list)
            object_prediction_list: list[Any] = []
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


def run_sahi_prediction_for_image(
    image_path: Path,
    get_sliced_prediction: Any,
    sahi_model: Any,
    class_names: Sequence[str],
    class_score_thresholds: dict[str, float],
    score_threshold: float,
    slice_height: Optional[int],
    slice_width: Optional[int],
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    perform_standard_pred: bool,
    postprocess_type: str,
    postprocess_match_metric: str,
    postprocess_match_threshold: float,
    postprocess_class_agnostic: bool,
) -> dict[str, Any]:
    prediction_result = get_sliced_prediction(
        str(image_path),
        detection_model=sahi_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        perform_standard_pred=perform_standard_pred,
        postprocess_type=postprocess_type,
        postprocess_match_metric=postprocess_match_metric,
        postprocess_match_threshold=postprocess_match_threshold,
        postprocess_class_agnostic=postprocess_class_agnostic,
        verbose=0,
    )

    pred_boxes_list: list[list[float]] = []
    pred_scores_list: list[float] = []
    pred_cls_list: list[int] = []
    for obj in prediction_result.object_prediction_list:
        x1, y1, x2, y2 = obj.bbox.to_xyxy()
        pred_boxes_list.append([float(x1), float(y1), float(x2), float(y2)])
        pred_scores_list.append(float(obj.score.value))
        pred_cls_list.append(int(obj.category.id))

    pred_boxes = np.asarray(pred_boxes_list, dtype=np.float32).reshape((-1, 4)) if pred_boxes_list else np.zeros((0, 4), dtype=np.float32)
    pred_scores = np.asarray(pred_scores_list, dtype=np.float32) if pred_scores_list else np.zeros((0,), dtype=np.float32)
    pred_cls = np.asarray(pred_cls_list, dtype=np.int64) if pred_cls_list else np.zeros((0,), dtype=np.int64)
    pred_cls = normalize_model_class_ids(pred_cls, class_names)

    keep = per_class_keep_mask(
        pred_scores,
        pred_cls,
        class_names,
        class_score_thresholds,
        score_threshold,
    )
    kept_boxes = pred_boxes[keep]
    kept_scores = pred_scores[keep]
    kept_cls = pred_cls[keep]
    return {
        "pred_boxes": pred_boxes,
        "pred_scores": pred_scores,
        "pred_cls": pred_cls,
        "kept_boxes": kept_boxes,
        "kept_scores": kept_scores,
        "kept_cls": kept_cls,
    }


def iou_matrix(boxes1: Any, boxes2: Any) -> Any:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    b1 = boxes1.astype(np.float32)
    b2 = boxes2.astype(np.float32)
    x11, y11, x12, y12 = b1[:, 0:1], b1[:, 1:2], b1[:, 2:3], b1[:, 3:4]
    x21, y21, x22, y22 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]
    ix1 = np.maximum(x11, x21)
    iy1 = np.maximum(y11, y21)
    ix2 = np.minimum(x12, x22)
    iy2 = np.minimum(y12, y22)
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area1 = np.maximum(0.0, x12 - x11) * np.maximum(0.0, y12 - y11)
    area2 = np.maximum(0.0, x22 - x21) * np.maximum(0.0, y22 - y21)
    union = area1 + area2 - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def box_area_xyxy(box: Any) -> float:
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_ios_xyxy(box1: Any, box2: Any) -> float:
    x11, y11, x12, y12 = [float(v) for v in box1.tolist()]
    x21, y21, x22, y22 = [float(v) for v in box2.tolist()]
    inter_x1 = max(x11, x21)
    inter_y1 = max(y11, y21)
    inter_x2 = min(x12, x22)
    inter_y2 = min(y12, y22)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    smaller = min(box_area_xyxy(box1), box_area_xyxy(box2))
    return inter / smaller if smaller > 0 else 0.0


def box_iou_xyxy(box1: Any, box2: Any) -> float:
    iou = iou_matrix(box1.reshape(1, 4), box2.reshape(1, 4))
    return float(iou[0, 0]) if iou.size else 0.0


def box_area_ratio(box1: Any, box2: Any) -> float:
    area1 = box_area_xyxy(box1)
    area2 = box_area_xyxy(box2)
    larger = max(area1, area2)
    smaller = min(area1, area2)
    return smaller / larger if larger > 0 else 0.0


def suppress_cross_class_duplicates(
    pred_boxes: Any,
    pred_scores: Any,
    pred_cls: Any,
    class_names: Sequence[str],
    ios_threshold: float,
    iou_threshold: float,
    area_ratio_threshold: float,
) -> tuple[Any, Any, Any, int]:
    if len(pred_boxes) <= 1:
        return pred_boxes, pred_scores, pred_cls, 0

    keep = np.ones((len(pred_boxes),), dtype=bool)
    order = np.argsort(-pred_scores)
    suppressed = 0

    # Match the scalar helpers' float64 area/IOS arithmetic. Keep float32 IoU
    # below, as in the original implementation, including threshold boundaries.
    boxes64 = np.asarray(pred_boxes, dtype=np.float64)
    areas = np.maximum(0.0, boxes64[:, 2] - boxes64[:, 0]) * np.maximum(0.0, boxes64[:, 3] - boxes64[:, 1])

    for pos_i, i in enumerate(order):
        if not keep[i]:
            continue
        candidates = order[pos_i + 1:]
        candidates = candidates[keep[candidates] & (pred_cls[candidates] != pred_cls[i])]
        if not len(candidates):
            continue
        smaller = np.minimum(areas[i], areas[candidates])
        larger = np.maximum(areas[i], areas[candidates])
        ratios = np.divide(smaller, larger, out=np.zeros_like(smaller), where=larger > 0)
        candidates = candidates[ratios >= area_ratio_threshold]
        if not len(candidates):
            continue
        left_top = np.maximum(boxes64[i, :2], boxes64[candidates, :2])
        right_bottom = np.minimum(boxes64[i, 2:], boxes64[candidates, 2:])
        wh = np.maximum(0.0, right_bottom - left_top)
        intersections = wh[:, 0] * wh[:, 1]
        smaller = np.minimum(areas[i], areas[candidates])
        ios = np.divide(intersections, smaller, out=np.zeros_like(smaller), where=smaller > 0)
        candidates = candidates[ios >= ios_threshold]
        if len(candidates):
            ious = iou_matrix(pred_boxes[i:i + 1], pred_boxes[candidates])[0]
            duplicates = candidates[ious >= iou_threshold]
            keep[duplicates] = False
            suppressed += len(duplicates)

    return pred_boxes[keep], pred_scores[keep], pred_cls[keep], suppressed


def count_class_predictions(pred_cls: Any, class_names: Sequence[str]) -> dict[str, int]:
    counts = {str(name): 0 for name in class_names}
    for cls_idx in pred_cls.tolist():
        if 0 <= int(cls_idx) < len(class_names):
            counts[class_names[int(cls_idx)]] += 1
    return counts


def leucocyte_score(leu_count: int) -> int:
    if leu_count < 10:
        return -1
    if leu_count <= 25:
        return 0
    if leu_count <= 50:
        return 1
    return 2


def squamous_epithelial_score(epi_count: int) -> int:
    if epi_count < 10:
        return 0
    if epi_count <= 25:
        return -1
    return -2


def classify_quality_from_counts(leu_count: int, epi_count: int) -> tuple[int, int, int, int]:
    leu_score = leucocyte_score(leu_count)
    epi_score = squamous_epithelial_score(epi_count)
    total_score = leu_score + epi_score

    if total_score >= 1 and epi_count < 10:
        label_id = 1
    elif total_score <= -1:
        label_id = 3
    else:
        label_id = 2
    return label_id, leu_score, epi_score, total_score


def overall_qa_label(processed_fovs: list[FOVRecord], rule_name: str) -> str:
    completed = [fov for fov in processed_fovs if fov.result is not None]
    if not completed:
        return "Pending"
    if rule_name == "Pending rule":
        return "Rule pending"

    labels = [fov.result.predicted_label for fov in completed if fov.result is not None]
    if rule_name == "Worst FOV":
        if "Not Qualified" in labels:
            return "Not Qualified"
        if "Partially Qualified" in labels:
            return "Partially Qualified"
        return "Qualified"

    counts = Counter(labels)
    return min(
        counts.items(),
        key=lambda item: (-item[1], -SEVERITY.get(item[0], 0)),
    )[0]


def _build_runtime(
    checkpoint_path: Path,
    model_class: str,
    model_resolution: Optional[int],
    class_names: Sequence[str],
    class_score_thresholds: Dict[str, float],
) -> InferenceRuntime:
    ensure_inference_deps()
    configure_torch_inference_backend()
    _, get_sliced_prediction, _ = import_sahi()

    num_classes = len(class_names)
    model = load_model_for_fov(
        model_class,
        checkpoint_path,
        model_resolution,
        num_classes,
        class_names,
    )
    sahi_model = build_sahi_model(model, class_names, DEFAULT_SCORE_FLOOR)
    return InferenceRuntime(
        get_sliced_prediction=get_sliced_prediction,
        sahi_model=sahi_model,
        class_names=list(class_names),
        class_score_thresholds=dict(class_score_thresholds),
        model_class=model_class,
        model_resolution=model_resolution,
    )


def build_runtime_for_checkpoint(checkpoint_path: Path) -> InferenceRuntime:
    """Build the final downstream runtime directly from checkpoint metadata."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    model_class, model_resolution, class_names, class_score_thresholds = inspect_checkpoint(checkpoint_path)
    return _build_runtime(
        checkpoint_path=checkpoint_path,
        model_class=model_class,
        model_resolution=model_resolution,
        class_names=class_names,
        class_score_thresholds=class_score_thresholds,
    )


def build_runtime(session: SampleSession) -> InferenceRuntime:
    return _build_runtime(
        checkpoint_path=session.checkpoint_path,
        model_class=session.model_class,
        model_resolution=session.model_resolution,
        class_names=session.class_names,
        class_score_thresholds=session.class_score_thresholds,
    )


def downstream_inference_settings() -> Dict[str, Any]:
    """Return the locked settings shared by the UI and batch evaluation."""
    return {
        "score_floor": DEFAULT_SCORE_FLOOR,
        "fallback_score_threshold": DEFAULT_SCORE_THRESHOLD,
        "class_score_thresholds": dict(CLASS_SCORE_THRESHOLDS),
        "slice_height": DEFAULT_SLICE_HEIGHT,
        "slice_width": DEFAULT_SLICE_WIDTH,
        "overlap_height_ratio": DEFAULT_OVERLAP_HEIGHT_RATIO,
        "overlap_width_ratio": DEFAULT_OVERLAP_WIDTH_RATIO,
        "perform_standard_prediction": DEFAULT_PERFORM_STANDARD_PRED,
        "postprocess": {
            "type": DEFAULT_POSTPROCESS_TYPE,
            "match_metric": DEFAULT_POSTPROCESS_MATCH_METRIC,
            "match_threshold": DEFAULT_POSTPROCESS_MATCH_THRESHOLD,
            "class_agnostic": DEFAULT_POSTPROCESS_CLASS_AGNOSTIC,
        },
        "cross_class_duplicate_suppression": {
            "enabled": True,
            "ios_threshold": CROSS_CLASS_DUPLICATE_IOS_THRESHOLD,
            "iou_threshold": CROSS_CLASS_DUPLICATE_IOU_THRESHOLD,
            "area_ratio_threshold": CROSS_CLASS_DUPLICATE_AREA_RATIO_THRESHOLD,
        },
    }


def run_inference_on_image(image_path: Path, runtime: InferenceRuntime) -> FOVInferenceResult:
    if np is None:
        raise ImportError("numpy is required.")

    if torch is not None:
        inference_context = torch.inference_mode()
    else:
        inference_context = nullcontext()

    with inference_context:
        prediction = run_sahi_prediction_for_image(
            image_path=image_path,
            get_sliced_prediction=runtime.get_sliced_prediction,
            sahi_model=runtime.sahi_model,
            class_names=runtime.class_names,
            class_score_thresholds=runtime.class_score_thresholds,
            score_threshold=DEFAULT_SCORE_THRESHOLD,
            slice_height=DEFAULT_SLICE_HEIGHT,
            slice_width=DEFAULT_SLICE_WIDTH,
            overlap_height_ratio=DEFAULT_OVERLAP_HEIGHT_RATIO,
            overlap_width_ratio=DEFAULT_OVERLAP_WIDTH_RATIO,
            perform_standard_pred=DEFAULT_PERFORM_STANDARD_PRED,
            postprocess_type=DEFAULT_POSTPROCESS_TYPE,
            postprocess_match_metric=DEFAULT_POSTPROCESS_MATCH_METRIC,
            postprocess_match_threshold=DEFAULT_POSTPROCESS_MATCH_THRESHOLD,
            postprocess_class_agnostic=DEFAULT_POSTPROCESS_CLASS_AGNOSTIC,
        )
    pred_boxes = prediction["pred_boxes"]
    kept_boxes = prediction["kept_boxes"]
    kept_scores = prediction["kept_scores"]
    kept_cls = prediction["kept_cls"]
    n_kept_before_duplicate_suppression = int(len(kept_boxes))

    kept_boxes, kept_scores, kept_cls, n_cross_class_duplicates_suppressed = suppress_cross_class_duplicates(
        kept_boxes,
        kept_scores,
        kept_cls,
        runtime.class_names,
        CROSS_CLASS_DUPLICATE_IOS_THRESHOLD,
        CROSS_CLASS_DUPLICATE_IOU_THRESHOLD,
        CROSS_CLASS_DUPLICATE_AREA_RATIO_THRESHOLD,
    )

    count_map = count_class_predictions(kept_cls, runtime.class_names)
    leu_count = int(count_map.get("Leucocyte", 0))
    epi_count = int(count_map.get("Squamous Epithelial Cell", 0))
    predicted_label_id, leu_score, epi_score, total_score = classify_quality_from_counts(leu_count, epi_count)

    return FOVInferenceResult(
        predicted_label_id=predicted_label_id,
        predicted_label=DOWNSTREAM_LABELS[predicted_label_id],
        n_leucocyte=leu_count,
        n_squamous_epithelial_cell=epi_count,
        leucocyte_score=leu_score,
        squamous_epithelial_score=epi_score,
        total_quality_score=total_score,
        n_predictions_raw=int(len(pred_boxes)),
        n_predictions_kept_before_duplicate_suppression=n_kept_before_duplicate_suppression,
        n_cross_class_duplicates_suppressed=int(n_cross_class_duplicates_suppressed),
        n_predictions_kept=int(len(kept_boxes)),
        pred_boxes=[tuple(float(value) for value in box.tolist()) for box in kept_boxes],
        pred_scores=[float(value) for value in kept_scores.tolist()],
        pred_cls=[int(value) for value in kept_cls.tolist()],
    )


def render_result_overlay(image_path: Path, result: FOVInferenceResult, class_names: list[str], *, show_quality_label: bool = True) -> Any:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise ImportError("Pillow is required.")

    with Image.open(image_path) as raw_img:
        img = raw_img.convert("RGB")

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    colors = class_color_map(class_names)

    for box, score, cls_idx in zip(result.pred_boxes, result.pred_scores, result.pred_cls):
        color = colors.get(int(cls_idx), (255, 0, 0))
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = score_to_text(float(score))
        text_w, text_h = _measure_text(draw, text, font)
        tx, ty = _score_anchor(box, img.width, img.height, text_w, text_h)
        draw_text_with_outline(draw, (tx, ty), text, fill=color, font=font)

    info_lines = [
        f"Predicted: {result.predicted_label}",
        f"Leucocytes: {result.n_leucocyte}",
        f"Squamous epithelial: {result.n_squamous_epithelial_cell}",
        f"Total score: {result.total_quality_score}",
    ]
    legend_lines = [f"{class_names[idx]} = prediction" for idx in range(len(class_names))]
    if not show_quality_label:
        info_lines = [
            f"Leucocyte: {result.n_leucocyte}",
            f"Squamous epithelial: {result.n_squamous_epithelial_cell}",
        ]
    all_lines = info_lines + legend_lines
    pad = 6
    line_h = max(_measure_text(draw, "Ag", font)[1], 10) + 2
    panel_w = max((_measure_text(draw, line, font)[0] for line in all_lines), default=0) + 30
    panel_h = len(all_lines) * line_h + pad * 2
    draw.rectangle([6, 6, 6 + panel_w, 6 + panel_h], fill=(0, 0, 0))
    y = 6 + pad
    for line in info_lines:
        draw_text_with_outline(draw, (12, y), line, fill=(255, 255, 255), font=font)
        y += line_h
    for idx, line in enumerate(legend_lines):
        color = colors.get(idx, (255, 0, 0))
        draw.rectangle([12, y + 2, 24, y + line_h - 2], outline=color, width=2)
        draw_text_with_outline(draw, (30, y), line, fill=(255, 255, 255), font=font)
        y += line_h

    return img
