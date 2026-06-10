from __future__ import annotations

import argparse
import importlib.util
import queue
import re
import sys
import threading
import time
import traceback
import tkinter as tk
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

try:
    import numpy as np
except Exception:
    np = None  # type: ignore[assignment]

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk
except Exception:
    Image = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SOLO_DIR = PROJECT_ROOT / "SOLO_Supervised_RFDETR"
for import_path in (PROJECT_ROOT, SOLO_DIR):
    import_str = str(import_path)
    if import_str not in sys.path:
        sys.path.insert(0, import_str)

import DownstreamEvaluation as downstream_eval
import EvalRFDETR_SAHI_FOV_Local as fov_eval
import EvalRFDETR_SOLO_LocalOnly as local_eval


QA_COLORS = {
    "Qualified": "#2e7d32",
    "Partially qualified": "#d97706",
    "Not qualified": "#c62828",
    "Pending": "#475569",
    "Processing": "#2563eb",
    "Error": "#7f1d1d",
    "Rule pending": "#64748b",
}

SLIDE_BG = "#111827"
APP_BG = "#f3f4f6"
CARD_BG = "#ffffff"
TEXT_PRIMARY = "#111827"
TEXT_SECONDARY = "#4b5563"

SEVERITY = {
    "Qualified": 0,
    "Partially qualified": 1,
    "Not qualified": 2,
}

OVERALL_RULES = (
    "Worst FOV",
    "Majority vote",
    "Pending rule",
)

HEATMAP_FILTERS = (
    "All",
    "Qualified",
    "Partially qualified",
    "Not qualified",
)

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
SAMPLE_RE = re.compile(r"(Sample\s*\d+|Sample\d+)", re.IGNORECASE)
COORD_RE = re.compile(r"BF\.(\d+)_(\d+)", re.IGNORECASE)

DEFAULT_SCORE_FLOOR = 0.001
DEFAULT_SCORE_THRESHOLD = 0.30
DEFAULT_SLICE_HEIGHT = 672
DEFAULT_SLICE_WIDTH = 672
DEFAULT_OVERLAP_HEIGHT_RATIO = 0.20
DEFAULT_OVERLAP_WIDTH_RATIO = 0.20
DEFAULT_PERFORM_STANDARD_PRED = False
DEFAULT_POSTPROCESS_TYPE = "GREEDYNMM"
DEFAULT_POSTPROCESS_MATCH_METRIC = "IOU"
DEFAULT_POSTPROCESS_MATCH_THRESHOLD = 0.50
DEFAULT_POSTPROCESS_CLASS_AGNOSTIC = False
SELECTED_PREVIEW_SIZE = 220

DEFAULT_CHECKPOINT = Path(
    downstream_eval.DOWNSTREAM_PRESETS[downstream_eval.ACTIVE_PRESET]["checkpoint"]
)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _blend_hex_color(low_color: str, high_color: str, fraction: float) -> str:
    fraction = max(0.0, min(1.0, fraction))
    low = _hex_to_rgb(low_color)
    high = _hex_to_rgb(high_color)
    blended = tuple(int(round(low[idx] + (high[idx] - low[idx]) * fraction)) for idx in range(3))
    return f"#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}"


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


def overall_qa_label(processed_fovs: list[FOVRecord], rule_name: str) -> str:
    completed = [fov for fov in processed_fovs if fov.result is not None]
    if not completed:
        return "Pending"
    if rule_name == "Pending rule":
        return "Rule pending"

    labels = [fov.result.predicted_label for fov in completed if fov.result is not None]
    if rule_name == "Worst FOV":
        if "Not qualified" in labels:
            return "Not qualified"
        if "Partially qualified" in labels:
            return "Partially qualified"
        return "Qualified"

    counts = Counter(labels)
    return min(
        counts.items(),
        key=lambda item: (-item[1], -SEVERITY.get(item[0], 0)),
    )[0]


def make_placeholder_image(size: tuple[int, int], text: str, accent: str) -> Image.Image:
    if Image is None or ImageDraw is None:
        raise ImportError("Pillow is required for the UI.")
    image = Image.new("RGB", size, "#d1d5db")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((4, 4, size[0] - 4, size[1] - 4), radius=10, fill="#e5e7eb", outline=accent, width=4)
    draw.text((size[0] // 2, size[1] // 2), text, fill="#374151", anchor="mm")
    return image


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
        for path in fov_eval.discover_images([sample_dir])
        if path.suffix.lower() in IMAGE_EXTENSIONS and "patch" not in str(path).lower()
    ]

    if not images:
        raise RuntimeError(f"No microscope images were found under {sample_dir}")
    return images


def inspect_checkpoint(checkpoint_path: Path) -> tuple[str, int | None, list[str], dict[str, float]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    model_class = local_eval.infer_model_class(checkpoint_path.parent, checkpoint_path)
    model_resolution = local_eval.infer_model_resolution(checkpoint_path)
    ckpt_num_classes, ckpt_class_names = fov_eval.infer_checkpoint_runtime_metadata(checkpoint_path)

    class_names = list(ckpt_class_names or fov_eval.DEFAULT_CLASS_NAMES)
    if ckpt_num_classes is not None and ckpt_num_classes > len(class_names):
        class_names = list(class_names) + [f"Class {idx}" for idx in range(len(class_names), ckpt_num_classes)]

    class_score_thresholds = {
        name: float(fov_eval.CLASS_SCORE_THRESHOLDS.get(name, DEFAULT_SCORE_THRESHOLD))
        for name in class_names
    }
    return model_class, model_resolution, class_names, class_score_thresholds


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

    local_eval.patch_transformers_torch_compat()
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
    spec.loader.exec_module(module)


def _patch_transformers_pruning_compat() -> None:
    try:
        import torch
        import transformers.pytorch_utils as pytorch_utils
    except Exception:
        return

    if hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
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


def build_sample_session(checkpoint_path: Path, sample_dir: Path) -> SampleSession:
    model_class, model_resolution, class_names, class_score_thresholds = inspect_checkpoint(checkpoint_path)
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
        class_names=class_names,
        class_score_thresholds=class_score_thresholds,
        fovs=fovs,
    )


def _read_image_size(image_path: Path) -> tuple[int, int]:
    try:
        with Image.open(image_path) as raw_img:
            return max(1, int(raw_img.width)), max(1, int(raw_img.height))
    except Exception:
        return 1, 1


def configure_torch_inference_backend() -> None:
    try:
        import torch
    except Exception:
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


def build_runtime(session: SampleSession) -> InferenceRuntime:
    if np is None:
        raise ImportError("numpy is required.")
    configure_torch_inference_backend()
    fov_eval.ensure_deps()
    _, get_sliced_prediction, _ = fov_eval.import_sahi()
    force_repo_local_rfdetr()

    num_classes = len(session.class_names)
    model = fov_eval.load_model_for_fov(
        session.model_class,
        session.checkpoint_path,
        session.model_resolution,
        num_classes,
        session.class_names,
    )
    sahi_model = fov_eval.build_sahi_model(model, session.class_names, DEFAULT_SCORE_FLOOR)
    return InferenceRuntime(
        get_sliced_prediction=get_sliced_prediction,
        sahi_model=sahi_model,
        class_names=list(session.class_names),
        class_score_thresholds=dict(session.class_score_thresholds),
        model_class=session.model_class,
        model_resolution=session.model_resolution,
    )


def run_inference_on_image(image_path: Path, runtime: InferenceRuntime) -> FOVInferenceResult:
    if np is None:
        raise ImportError("numpy is required.")

    try:
        import torch

        inference_context = torch.inference_mode()
    except Exception:
        inference_context = nullcontext()

    with inference_context:
        prediction = fov_eval.run_sahi_prediction_for_image(
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

    kept_boxes, kept_scores, kept_cls, n_cross_class_duplicates_suppressed = downstream_eval.suppress_cross_class_duplicates(
        kept_boxes,
        kept_scores,
        kept_cls,
        runtime.class_names,
        downstream_eval.CROSS_CLASS_DUPLICATE_IOS_THRESHOLD,
        downstream_eval.CROSS_CLASS_DUPLICATE_IOU_THRESHOLD,
        downstream_eval.CROSS_CLASS_DUPLICATE_AREA_RATIO_THRESHOLD,
    )

    count_map = downstream_eval.count_class_predictions(kept_cls, runtime.class_names)
    leu_count = int(count_map.get("Leucocyte", 0))
    epi_count = int(count_map.get("Squamous Epithelial Cell", 0))
    predicted_label_id, leu_score, epi_score, total_score = downstream_eval.classify_quality_from_counts(leu_count, epi_count)

    return FOVInferenceResult(
        predicted_label_id=predicted_label_id,
        predicted_label=downstream_eval.DOWNSTREAM_LABELS[predicted_label_id],
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


def render_result_overlay(image_path: Path, result: FOVInferenceResult, class_names: list[str]) -> Image.Image:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise ImportError("Pillow is required.")

    with Image.open(image_path) as raw_img:
        img = raw_img.convert("RGB")

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    colors = fov_eval.class_color_map(class_names)

    for box, score, cls_idx in zip(result.pred_boxes, result.pred_scores, result.pred_cls):
        color = colors.get(int(cls_idx), (255, 0, 0))
        x1, y1, x2, y2 = [int(round(value)) for value in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = fov_eval.score_to_text(float(score))
        text_w, text_h = fov_eval._measure_text(draw, text, font)
        tx, ty = fov_eval._score_anchor(box, img.width, img.height, text_w, text_h)
        fov_eval.draw_text_with_outline(draw, (tx, ty), text, fill=color, font=font)

    info_lines = [
        f"Predicted: {result.predicted_label}",
        f"Leucocyte count: {result.n_leucocyte}",
        f"Sq. epithelial count: {result.n_squamous_epithelial_cell}",
        f"Quality score: {result.total_quality_score}",
    ]
    legend_lines = [f"{class_names[idx]} = prediction" for idx in range(len(class_names))]
    all_lines = info_lines + legend_lines
    pad = 6
    line_h = max(fov_eval._measure_text(draw, "Ag", font)[1], 10) + 2
    panel_w = max((fov_eval._measure_text(draw, line, font)[0] for line in all_lines), default=0) + 30
    panel_h = len(all_lines) * line_h + pad * 2
    draw.rectangle([6, 6, 6 + panel_w + pad * 2, 6 + panel_h], fill=(0, 0, 0))

    for idx, line in enumerate(info_lines):
        y = 6 + pad + idx * line_h
        fov_eval.draw_text_with_outline(draw, (12, y), line, fill=(255, 255, 255), font=font)
    for idx, line in enumerate(legend_lines, start=len(info_lines)):
        y = 6 + pad + idx * line_h
        color = colors.get(idx - len(info_lines), (255, 0, 0))
        draw.rectangle([12, y + 2, 24, y + line_h - 2], outline=color, width=2)
        fov_eval.draw_text_with_outline(draw, (30, y), line, fill=(255, 255, 255), font=font)

    return img


def check_dependencies() -> None:
    if Image is None or ImageDraw is None or ImageFont is None or ImageOps is None or ImageTk is None:
        raise ImportError("Pillow is required for the UI.")
    if np is None:
        raise ImportError("numpy is required.")


class QAWorkflowUI:
    def __init__(
        self,
        root: tk.Tk,
        initial_checkpoint: Path | None = None,
        initial_sample_folder: Path | None = None,
        auto_load: bool = False,
    ) -> None:
        check_dependencies()

        self.root = root
        self.sample: SampleSession | None = None
        self.samples: list[SampleSession] = []
        self.current_sample_index = 0
        self.current_processing_sample_index: int | None = None
        self.selected_fov: FOVRecord | None = None

        self.job_queue: queue.Queue[tuple[int, str, Any]] = queue.Queue()
        self.current_job_id = 0
        self.worker_thread: threading.Thread | None = None
        self.worker_cancel_event: threading.Event | None = None
        self.worker_pause_event: threading.Event | None = None
        self.current_processing_index: int | None = None
        self.is_running = False
        self.is_paused = False

        self.slide_geometry = (0.0, 0.0, 0.0)
        self.ranking_geometry = (0.0, 0.0, 0.0)
        self.slide_zoom = 1.0
        self._suppress_ranking_select = False
        self.photo_cache: dict[tuple[str, int, int], ImageTk.PhotoImage] = {}
        self._canvas_images: list[ImageTk.PhotoImage] = []
        self.preview_photo: ImageTk.PhotoImage | None = None

        self.checkpoint_path = tk.StringVar(value=str(initial_checkpoint or (DEFAULT_CHECKPOINT if DEFAULT_CHECKPOINT.exists() else "")))
        self.sample_folder = tk.StringVar(value=str(initial_sample_folder or ""))
        self.sample_nav_var = tk.StringVar(value="No samples loaded")
        self.overall_rule = tk.StringVar(value=OVERALL_RULES[0])
        self.slide_zoom_label_var = tk.StringVar(value="100%")
        self.heatmap_filter = tk.StringVar(value=HEATMAP_FILTERS[0])
        self.live_slide_updates = tk.BooleanVar(value=True)
        self.live_heatmap_updates = tk.BooleanVar(value=False)

        self.model_class_var = tk.StringVar(value="-")
        self.resolution_var = tk.StringVar(value="-")
        self.class_names_var = tk.StringVar(value="-")

        self.present_var = tk.StringVar(value="0")
        self.completed_var = tk.StringVar(value="0 / 0")
        self.pending_var = tk.StringVar(value="0")
        self.qualified_var = tk.StringVar(value="0")
        self.partial_var = tk.StringVar(value="0")
        self.not_qualified_var = tk.StringVar(value="0")
        self.leucocyte_total_var = tk.StringVar(value="0")
        self.squamous_total_var = tk.StringVar(value="0")
        self.inference_time_var = tk.StringVar(value="-")

        self.fov_name_var = tk.StringVar(value="-")
        self.fov_position_var = tk.StringVar(value="-")
        self.fov_stage_var = tk.StringVar(value="-")
        self.fov_qa_var = tk.StringVar(value="-")
        self.fov_leucocyte_var = tk.StringVar(value="-")
        self.fov_squamous_var = tk.StringVar(value="-")
        self.fov_predictions_var = tk.StringVar(value="-")
        self.fov_score_var = tk.StringVar(value="-")

        self.root.title("Microscope QA Workflow")
        self.root.geometry("1540x940")
        self.root.configure(bg=APP_BG)

        self._configure_styles()
        self._build_ui()
        self._bind_events()

        self._update_button_states()
        self._update_status("Select checkpoint and sample folder.")
        self.refresh_ui()
        self.root.after(100, self._poll_worker_queue)

        if auto_load and self.checkpoint_path.get().strip() and self.sample_folder.get().strip():
            self.load_selected_sample(auto_start=True)

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("App.TFrame", background=APP_BG)
        style.configure("Card.TFrame", background=CARD_BG)
        style.configure("Card.TLabelframe", background=CARD_BG, borderwidth=1, relief="solid")
        style.configure("Card.TLabelframe.Label", background=CARD_BG, foreground=TEXT_PRIMARY, font=("Segoe UI", 11, "bold"))
        style.configure("Header.TLabel", background=APP_BG, foreground=TEXT_PRIMARY, font=("Segoe UI", 18, "bold"))
        style.configure("Muted.TLabel", background=CARD_BG, foreground=TEXT_SECONDARY, font=("Segoe UI", 10))
        style.configure("Value.TLabel", background=CARD_BG, foreground=TEXT_PRIMARY, font=("Segoe UI", 11, "bold"))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"))

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="App.TFrame", padding=(18, 14))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        ttk.Label(header, text="Clinical QA Workflow", style="Header.TLabel").grid(row=0, column=0, sticky="w")

        self.status_label = tk.Label(
            header,
            text="",
            bg=APP_BG,
            fg=TEXT_SECONDARY,
            font=("Segoe UI", 11),
            padx=8,
            pady=4,
        )
        self.status_label.grid(row=0, column=1, sticky="w", padx=(18, 0))

        self.overall_badge = tk.Label(
            header,
            text="Awaiting QA",
            bg=QA_COLORS["Pending"],
            fg="white",
            font=("Segoe UI", 12, "bold"),
            padx=12,
            pady=6,
        )
        self.overall_badge.grid(row=0, column=2, sticky="e")

        body = ttk.Frame(self.root, style="App.TFrame", padding=(18, 0, 18, 18))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=65)
        body.columnconfigure(1, weight=35)
        body.columnconfigure(2, weight=0)
        body.rowconfigure(0, weight=1)

        slide_frame = ttk.Labelframe(body, text="Slide Layout", style="Card.TLabelframe", padding=12)
        slide_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        slide_frame.rowconfigure(0, weight=0)
        slide_frame.rowconfigure(1, weight=4)
        slide_frame.rowconfigure(2, weight=1)
        slide_frame.columnconfigure(0, weight=1)

        slide_toolbar = ttk.Frame(slide_frame, style="Card.TFrame")
        slide_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        slide_toolbar.columnconfigure(4, weight=1)
        ttk.Button(slide_toolbar, text="Fit", command=self.fit_slide_to_view).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(slide_toolbar, text="-", width=3, command=lambda: self.zoom_slide(0.8)).grid(row=0, column=1, padx=(0, 4))
        ttk.Button(slide_toolbar, text="+", width=3, command=lambda: self.zoom_slide(1.25)).grid(row=0, column=2, padx=(0, 8))
        tk.Label(
            slide_toolbar,
            textvariable=self.slide_zoom_label_var,
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            font=("Segoe UI", 10, "bold"),
            width=6,
            anchor="w",
        ).grid(row=0, column=3, sticky="w")

        slide_view = ttk.Frame(slide_frame, style="Card.TFrame")
        slide_view.grid(row=1, column=0, sticky="nsew")
        slide_view.rowconfigure(0, weight=1)
        slide_view.columnconfigure(0, weight=1)

        self.slide_canvas = tk.Canvas(slide_view, bg=SLIDE_BG, highlightthickness=0, relief="flat")
        self.slide_canvas.grid(row=0, column=0, sticky="nsew")
        self.slide_y_scrollbar = ttk.Scrollbar(slide_view, orient="vertical", command=self.slide_canvas.yview)
        self.slide_y_scrollbar.grid(row=0, column=1, sticky="ns")
        self.slide_x_scrollbar = ttk.Scrollbar(slide_view, orient="horizontal", command=self.slide_canvas.xview)
        self.slide_x_scrollbar.grid(row=1, column=0, sticky="ew")
        self.slide_canvas.configure(
            xscrollcommand=self.slide_x_scrollbar.set,
            yscrollcommand=self.slide_y_scrollbar.set,
        )

        ranking = ttk.Labelframe(slide_frame, text="FOV Ranking", style="Card.TLabelframe", padding=10)
        ranking.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        ranking.rowconfigure(0, weight=1)
        ranking.columnconfigure(0, weight=1)

        self.ranking_canvas = tk.Canvas(ranking, bg=SLIDE_BG, highlightthickness=0, relief="flat")
        self.ranking_canvas.grid(row=0, column=0, sticky="nsew")

        heatmap_panel = ttk.Labelframe(body, text="Heatmaps", style="Card.TLabelframe", padding=12)
        heatmap_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 16))
        heatmap_panel.columnconfigure(0, weight=1)
        heatmap_panel.rowconfigure(1, weight=1)
        heatmap_panel.rowconfigure(2, weight=1)

        heatmap_filter_bar = ttk.Frame(heatmap_panel, style="Card.TFrame")
        heatmap_filter_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        heatmap_filter_bar.columnconfigure(2, weight=1)
        tk.Label(
            heatmap_filter_bar,
            text="Heatmap FOVs",
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.heatmap_filter_combo = ttk.Combobox(
            heatmap_filter_bar,
            textvariable=self.heatmap_filter,
            values=HEATMAP_FILTERS,
            state="readonly",
            width=22,
        )
        self.heatmap_filter_combo.grid(row=0, column=1, sticky="w")

        leucocyte_panel = ttk.Frame(heatmap_panel, style="Card.TFrame")
        leucocyte_panel.grid(row=1, column=0, sticky="nsew", pady=(0, 12))
        leucocyte_panel.columnconfigure(0, weight=1)
        leucocyte_panel.rowconfigure(1, weight=1)
        tk.Label(
            leucocyte_panel,
            text="Leucocyte distribution",
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.leucocyte_heatmap_canvas = tk.Canvas(leucocyte_panel, bg="#111827", height=330, highlightthickness=0)
        self.leucocyte_heatmap_canvas.grid(row=1, column=0, sticky="nsew")

        epithelial_panel = ttk.Frame(heatmap_panel, style="Card.TFrame")
        epithelial_panel.grid(row=2, column=0, sticky="nsew")
        epithelial_panel.columnconfigure(0, weight=1)
        epithelial_panel.rowconfigure(1, weight=1)
        tk.Label(
            epithelial_panel,
            text="Epithelial-cell distribution",
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.epithelial_heatmap_canvas = tk.Canvas(epithelial_panel, bg="#111827", height=330, highlightthickness=0)
        self.epithelial_heatmap_canvas.grid(row=1, column=0, sticky="nsew")

        side_panel = ttk.Frame(body, style="App.TFrame")
        side_panel.grid(row=0, column=2, sticky="ns")
        side_panel.columnconfigure(0, weight=1)
        side_panel.rowconfigure(2, weight=0)

        controls = ttk.Labelframe(side_panel, text="Setup", style="Card.TLabelframe", padding=12)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)

        ttk.Label(controls, text="Checkpoint", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        checkpoint_row = ttk.Frame(controls, style="Card.TFrame")
        checkpoint_row.grid(row=1, column=0, sticky="ew", pady=(2, 8))
        checkpoint_row.columnconfigure(0, weight=1)
        self.checkpoint_entry = ttk.Entry(checkpoint_row, textvariable=self.checkpoint_path)
        self.checkpoint_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(checkpoint_row, text="Browse", command=self.select_checkpoint).grid(row=0, column=1, padx=(8, 0))

        ttk.Label(controls, text="Sample folders", style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        sample_row = ttk.Frame(controls, style="Card.TFrame")
        sample_row.grid(row=3, column=0, sticky="ew", pady=(2, 8))
        sample_row.columnconfigure(0, weight=1)
        self.sample_entry = ttk.Entry(sample_row, textvariable=self.sample_folder)
        self.sample_entry.grid(row=0, column=0, sticky="ew")
        ttk.Button(sample_row, text="Add", command=self.select_sample_folder).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(sample_row, text="Clear", command=self.clear_sample_folders).grid(row=0, column=2, padx=(6, 0))

        sample_nav = ttk.Frame(controls, style="Card.TFrame")
        sample_nav.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        sample_nav.columnconfigure(1, weight=1)
        self.previous_sample_button = ttk.Button(sample_nav, text="<", width=3, command=self.show_previous_sample)
        self.previous_sample_button.grid(row=0, column=0, sticky="w")
        tk.Label(
            sample_nav,
            textvariable=self.sample_nav_var,
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            font=("Segoe UI", 10, "bold"),
            anchor="center",
        ).grid(row=0, column=1, sticky="ew", padx=6)
        self.next_sample_button = ttk.Button(sample_nav, text=">", width=3, command=self.show_next_sample)
        self.next_sample_button.grid(row=0, column=2, sticky="e")

        ttk.Label(controls, text="Overall rule", style="Muted.TLabel").grid(row=5, column=0, sticky="w")
        self.rule_combo = ttk.Combobox(controls, textvariable=self.overall_rule, values=OVERALL_RULES, state="readonly")
        self.rule_combo.grid(row=6, column=0, sticky="ew", pady=(2, 12))

        button_row = ttk.Frame(controls, style="Card.TFrame")
        button_row.grid(row=7, column=0, sticky="ew")
        for column in range(4):
            button_row.columnconfigure(column, weight=1)

        self.load_button = ttk.Button(button_row, text="Load", style="Primary.TButton", command=lambda: self.load_selected_sample(auto_start=True))
        self.load_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.start_button = ttk.Button(button_row, text="Start", command=self.start_processing)
        self.start_button.grid(row=0, column=1, sticky="ew", padx=3)

        self.pause_button = ttk.Button(button_row, text="Pause", command=self.pause_processing)
        self.pause_button.grid(row=0, column=2, sticky="ew", padx=3)

        self.reset_button = ttk.Button(button_row, text="Reset", command=self.reset_processing)
        self.reset_button.grid(row=0, column=3, sticky="ew", padx=(6, 0))

        live_row = ttk.Frame(controls, style="Card.TFrame")
        live_row.grid(row=8, column=0, sticky="ew", pady=(10, 0))
        live_row.columnconfigure(0, weight=1)
        live_row.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            live_row,
            text="Live slide",
            variable=self.live_slide_updates,
            command=self.refresh_ui,
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            live_row,
            text="Live heatmaps",
            variable=self.live_heatmap_updates,
            command=self.draw_heatmaps,
        ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        summary = ttk.Labelframe(side_panel, text="Sample Summary", style="Card.TLabelframe", padding=12)
        summary.grid(row=1, column=0, sticky="ew", pady=(16, 0))
        summary.columnconfigure(0, weight=1)
        self._make_stat_row(summary, 0, "Present FOVs", self.present_var)
        self._make_stat_row(summary, 1, "QA complete", self.completed_var)
        self._make_stat_row(summary, 2, "Pending QA", self.pending_var)
        self._make_stat_row(summary, 3, "Qualified", self.qualified_var, color=QA_COLORS["Qualified"])
        self._make_stat_row(summary, 4, "Partially qualified", self.partial_var, color=QA_COLORS["Partially qualified"])
        self._make_stat_row(summary, 5, "Not qualified", self.not_qualified_var, color=QA_COLORS["Not qualified"])
        self._make_stat_row(summary, 6, "Leucocytes counted", self.leucocyte_total_var)
        self._make_stat_row(summary, 7, "Squamous cells counted", self.squamous_total_var)
        self._make_stat_row(summary, 8, "Inference time / FOV", self.inference_time_var)

        self.inference_progress = ttk.Progressbar(summary, maximum=1, mode="determinate")
        self.inference_progress.grid(row=9, column=0, sticky="ew", pady=(10, 0))

        detail = ttk.Labelframe(side_panel, text="Selected FOV", style="Card.TLabelframe", padding=12)
        detail.grid(row=2, column=0, sticky="ew", pady=(16, 0))
        detail.columnconfigure(0, weight=1)

        preview_frame = tk.Frame(detail, bg="#d1d5db", width=SELECTED_PREVIEW_SIZE, height=SELECTED_PREVIEW_SIZE)
        preview_frame.grid(row=0, column=0, pady=(0, 12))
        preview_frame.grid_propagate(False)
        self.preview_label = tk.Label(preview_frame, bg="#d1d5db", bd=0, highlightthickness=0)
        self.preview_label.place(relx=0.5, rely=0.5, anchor="center")

        self._make_detail_row(detail, 1, "FOV", self.fov_name_var)
        self._make_detail_row(detail, 2, "Position", self.fov_position_var)
        self._make_detail_row(detail, 3, "Stage", self.fov_stage_var)
        self._make_detail_row(detail, 4, "QA", self.fov_qa_var)
        self._make_detail_row(detail, 5, "Leucocytes", self.fov_leucocyte_var)
        self._make_detail_row(detail, 6, "Squamous cells", self.fov_squamous_var)

    def _bind_events(self) -> None:
        self.slide_canvas.bind("<Button-1>", self._on_canvas_click)
        self.slide_canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.slide_canvas.bind("<Configure>", lambda _event: self.draw_slide())
        self.slide_canvas.bind("<Control-MouseWheel>", self._on_slide_zoom_wheel)
        self.slide_canvas.bind("<MouseWheel>", self._on_slide_mouse_wheel)
        self.slide_canvas.bind("<Shift-MouseWheel>", self._on_slide_shift_mouse_wheel)
        self.ranking_canvas.bind("<Button-1>", self._on_ranking_click)
        self.ranking_canvas.bind("<Double-Button-1>", self._on_ranking_double_click)
        self.ranking_canvas.bind("<Configure>", lambda _event: self.draw_ranking())
        self.leucocyte_heatmap_canvas.bind("<Configure>", lambda _event: self.draw_heatmaps())
        self.epithelial_heatmap_canvas.bind("<Configure>", lambda _event: self.draw_heatmaps())
        self.rule_combo.bind("<<ComboboxSelected>>", lambda _event: self.refresh_ui())
        self.heatmap_filter_combo.bind("<<ComboboxSelected>>", lambda _event: self.draw_heatmaps())
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def zoom_slide(self, factor: float) -> None:
        self._set_slide_zoom(self.slide_zoom * factor)

    def fit_slide_to_view(self) -> None:
        self._set_slide_zoom(1.0)

    def _set_slide_zoom(self, zoom: float) -> None:
        self.slide_zoom = max(0.35, min(5.0, float(zoom)))
        self.slide_zoom_label_var.set(f"{int(round(self.slide_zoom * 100))}%")
        self.draw_slide()

    def _on_slide_zoom_wheel(self, event: tk.Event[tk.Canvas]) -> str:
        self.zoom_slide(1.15 if event.delta > 0 else 1 / 1.15)
        return "break"

    def _on_slide_mouse_wheel(self, event: tk.Event[tk.Canvas]) -> str:
        direction = -1 if event.delta > 0 else 1
        self.slide_canvas.yview_scroll(direction, "units")
        return "break"

    def _on_slide_shift_mouse_wheel(self, event: tk.Event[tk.Canvas]) -> str:
        direction = -1 if event.delta > 0 else 1
        self.slide_canvas.xview_scroll(direction, "units")
        return "break"

    def _make_stat_row(self, parent: ttk.Labelframe, row: int, label: str, variable: tk.StringVar, color: str | None = None) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        frame.columnconfigure(1, weight=1)

        accent = tk.Frame(frame, bg=color or "#d1d5db", width=6, height=28)
        accent.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        ttk.Label(frame, text=label, style="Muted.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(frame, textvariable=variable, style="Value.TLabel").grid(row=0, column=2, sticky="e")

    def _make_detail_row(self, parent: ttk.Labelframe, row: int, label: str, variable: tk.StringVar) -> None:
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=label, style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=variable, style="Value.TLabel").grid(row=0, column=1, sticky="e")

    def select_checkpoint(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select quality-assessment checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("All files", "*.*")],
        )
        if selected:
            self.checkpoint_path.set(selected)

    def select_sample_folder(self) -> None:
        selected_paths = self._ask_sample_folders()
        if selected_paths:
            self._add_sample_dirs(selected_paths)

    def _path_key(self, path: Path) -> str:
        try:
            return str(path.expanduser().resolve(strict=False)).lower()
        except OSError:
            return str(path.expanduser()).lower()

    def _add_sample_dirs(self, selected_paths: list[Path]) -> None:
        existing = self._selected_sample_dirs()
        existing_keys = {self._path_key(path) for path in existing}
        for selected_path in selected_paths:
            path = selected_path.expanduser()
            key = self._path_key(path)
            if key in existing_keys:
                continue
            existing.append(path)
            existing_keys.add(key)
        self.sample_folder.set(" ; ".join(str(path) for path in existing))

    def _default_sample_picker_parent(self) -> Path:
        existing = self._selected_sample_dirs()
        if existing:
            last_path = existing[-1].expanduser()
            if last_path.exists():
                return last_path.parent if last_path.is_dir() else last_path.parent

        raw_value = self.sample_folder.get().strip()
        if raw_value:
            first_value = re.split(r"[;\n]+", raw_value, maxsplit=1)[0].strip().strip('"')
            if first_value:
                first_path = Path(first_value).expanduser()
                if first_path.exists():
                    return first_path.parent if first_path.is_dir() else first_path.parent

        return Path.home()

    def _ask_sample_folders(self) -> list[Path]:
        selected_paths: list[Path] = []
        path_by_index: list[Path] = []

        dialog = tk.Toplevel(self.root)
        dialog.title("Add sample folders")
        dialog.configure(bg=CARD_BG)
        dialog.transient(self.root)
        dialog.geometry("640x520")
        dialog.minsize(520, 420)
        dialog.columnconfigure(0, weight=1)
        dialog.rowconfigure(2, weight=1)

        parent_var = tk.StringVar(value=str(self._default_sample_picker_parent()))
        filter_var = tk.StringVar()
        status_var = tk.StringVar(value="")

        parent_row = ttk.Frame(dialog, style="Card.TFrame", padding=(12, 12, 12, 4))
        parent_row.grid(row=0, column=0, sticky="ew")
        parent_row.columnconfigure(0, weight=1)
        parent_entry = ttk.Entry(parent_row, textvariable=parent_var)
        parent_entry.grid(row=0, column=0, sticky="ew")

        def browse_parent() -> None:
            initial_dir = parent_var.get().strip()
            if not initial_dir or not Path(initial_dir).expanduser().exists():
                initial_dir = str(Path.home())
            selected = filedialog.askdirectory(parent=dialog, title="Select parent folder", initialdir=initial_dir)
            if selected:
                parent_var.set(selected)
                populate_folder_list()

        ttk.Button(parent_row, text="Browse", command=browse_parent).grid(row=0, column=1, padx=(8, 0))

        filter_row = ttk.Frame(dialog, style="Card.TFrame", padding=(12, 4, 12, 4))
        filter_row.grid(row=1, column=0, sticky="ew")
        filter_row.columnconfigure(1, weight=1)
        ttk.Label(filter_row, text="Filter", style="Muted.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 8))
        filter_entry = ttk.Entry(filter_row, textvariable=filter_var)
        filter_entry.grid(row=0, column=1, sticky="ew")

        list_frame = ttk.Frame(dialog, style="Card.TFrame", padding=(12, 4, 12, 4))
        list_frame.grid(row=2, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        folder_listbox = tk.Listbox(
            list_frame,
            selectmode="extended",
            activestyle="dotbox",
            bg="#f8fafc",
            fg=TEXT_PRIMARY,
            selectbackground="#2563eb",
            selectforeground="white",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
            relief="flat",
            font=("Segoe UI", 10),
        )
        folder_listbox.grid(row=0, column=0, sticky="nsew")
        y_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=folder_listbox.yview)
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        folder_listbox.configure(yscrollcommand=y_scrollbar.set)

        status_row = ttk.Frame(dialog, style="Card.TFrame", padding=(12, 4, 12, 4))
        status_row.grid(row=3, column=0, sticky="ew")
        status_row.columnconfigure(0, weight=1)
        ttk.Label(status_row, textvariable=status_var, style="Muted.TLabel").grid(row=0, column=0, sticky="w")

        button_row = ttk.Frame(dialog, style="Card.TFrame", padding=(12, 8, 12, 12))
        button_row.grid(row=4, column=0, sticky="ew")
        button_row.columnconfigure(0, weight=1)

        def has_direct_fov_images(directory: Path) -> bool:
            try:
                return any(
                    path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and "patch" not in str(path).lower()
                    for path in directory.iterdir()
                )
            except OSError:
                return False

        def folder_sort_key(path: Path) -> list[Any]:
            return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]

        def update_status() -> None:
            selected_count = len(folder_listbox.curselection())
            total_count = len(path_by_index)
            status_var.set(f"{selected_count} selected / {total_count} folders")

        def populate_folder_list() -> None:
            folder_listbox.delete(0, tk.END)
            path_by_index.clear()
            parent_path = Path(parent_var.get().strip()).expanduser()
            filter_text = filter_var.get().strip().lower()
            if not parent_path.exists() or not parent_path.is_dir():
                status_var.set("Parent folder not found.")
                return

            if has_direct_fov_images(parent_path):
                path_by_index.append(parent_path)
                folder_listbox.insert(tk.END, f"[this sample folder] {parent_path.name}")
                folder_listbox.selection_set(0)
                update_status()
                return

            try:
                child_dirs = sorted(
                    (
                        path
                        for path in parent_path.iterdir()
                        if path.is_dir() and "patch" not in path.name.lower()
                    ),
                    key=folder_sort_key,
                )
            except OSError as exc:
                status_var.set(f"Could not read folder: {exc}")
                return

            for child_dir in child_dirs:
                if filter_text and filter_text not in child_dir.name.lower():
                    continue
                path_by_index.append(child_dir)
                folder_listbox.insert(tk.END, child_dir.name)
            update_status()

        def select_all() -> None:
            folder_listbox.selection_set(0, tk.END)
            update_status()

        def add_current_folder() -> None:
            parent_path = Path(parent_var.get().strip()).expanduser()
            if parent_path.exists() and parent_path.is_dir():
                selected_paths[:] = [parent_path]
            dialog.destroy()

        def add_selected_folders() -> None:
            selected_indices = [int(index) for index in folder_listbox.curselection()]
            selected_paths[:] = [path_by_index[index] for index in selected_indices if 0 <= index < len(path_by_index)]
            if not selected_paths:
                messagebox.showwarning("Add sample folders", "Select at least one sample folder.", parent=dialog)
                return
            dialog.destroy()

        ttk.Button(button_row, text="Select all", command=select_all).grid(row=0, column=0, sticky="w")
        ttk.Button(button_row, text="Add this folder", command=add_current_folder).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(button_row, text="Add selected", style="Primary.TButton", command=add_selected_folders).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(button_row, text="Cancel", command=dialog.destroy).grid(row=0, column=3, padx=(8, 0))

        filter_entry.bind("<KeyRelease>", lambda _event: populate_folder_list())
        parent_entry.bind("<Return>", lambda _event: populate_folder_list())
        folder_listbox.bind("<<ListboxSelect>>", lambda _event: update_status())
        folder_listbox.bind("<Double-Button-1>", lambda _event: add_selected_folders())

        populate_folder_list()
        dialog.grab_set()
        dialog.wait_window()
        return selected_paths

    def clear_sample_folders(self) -> None:
        self.sample_folder.set("")

    def _selected_sample_dirs(self) -> list[Path]:
        raw_value = self.sample_folder.get().strip()
        if not raw_value:
            return []

        sample_dirs: list[Path] = []
        seen: set[str] = set()
        for part in re.split(r"[;\n]+", raw_value):
            part = part.strip().strip('"')
            if not part:
                continue
            path = Path(part).expanduser()
            key = self._path_key(path)
            if key in seen:
                continue
            seen.add(key)
            sample_dirs.append(path)
        return sample_dirs

    def load_selected_sample(self, auto_start: bool) -> None:
        checkpoint = Path(self.checkpoint_path.get().strip()).expanduser()
        sample_dirs = self._selected_sample_dirs()
        if not sample_dirs:
            messagebox.showerror("Load sample", "Select at least one sample folder.")
            self._update_status("Sample load failed.")
            return

        sessions: list[SampleSession] = []
        try:
            for sample_dir in sample_dirs:
                sessions.append(build_sample_session(checkpoint, sample_dir))
        except Exception as exc:
            messagebox.showerror("Load sample", str(exc))
            self._update_status("Sample load failed.")
            return

        self._cancel_current_job()
        self.samples = sessions
        self.current_sample_index = 0
        self.sample = sessions[0] if sessions else None
        self.selected_fov = self.sample.fovs[0] if self.sample and self.sample.fovs else None
        self.current_processing_index = None
        self.current_processing_sample_index = None
        self.slide_zoom = 1.0
        self.slide_zoom_label_var.set("100%")
        self.photo_cache.clear()

        if self.sample is not None:
            self.model_class_var.set(self.sample.model_class)
            self.resolution_var.set(str(self.sample.model_resolution) if self.sample.model_resolution is not None else "Unknown")
            self.class_names_var.set(", ".join(self.sample.class_names))
            self.sample_folder.set(" ; ".join(str(session.sample_dir) for session in sessions))
        total_fovs = sum(len(session.fovs) for session in sessions)
        self._update_status(f"Loaded {len(sessions)} sample(s) with {total_fovs} FOVs.")
        self._update_sample_nav()
        self.refresh_ui()

        if auto_start:
            self.start_processing()

    def _set_current_sample_index(self, sample_index: int) -> None:
        if not self.samples:
            self.sample = None
            self.selected_fov = None
            self.current_sample_index = 0
            self._update_sample_nav()
            self.refresh_ui()
            return

        self.current_sample_index = max(0, min(sample_index, len(self.samples) - 1))
        self.sample = self.samples[self.current_sample_index]
        if self.selected_fov is None or all(self.selected_fov is not fov for fov in self.sample.fovs):
            self.selected_fov = self.sample.fovs[0] if self.sample.fovs else None
        self.photo_cache.clear()
        self._update_sample_nav()
        self.refresh_ui()

    def show_previous_sample(self) -> None:
        self._set_current_sample_index(self.current_sample_index - 1)

    def show_next_sample(self) -> None:
        self._set_current_sample_index(self.current_sample_index + 1)

    def _update_sample_nav(self) -> None:
        if not self.samples:
            self.sample_nav_var.set("No samples loaded")
            if hasattr(self, "previous_sample_button"):
                self.previous_sample_button.configure(state="disabled")
                self.next_sample_button.configure(state="disabled")
            return

        current = self.samples[self.current_sample_index]
        nav_text = f"{self.current_sample_index + 1}/{len(self.samples)} {current.sample_id}"
        if self.current_processing_sample_index is not None and 0 <= self.current_processing_sample_index < len(self.samples):
            processing = self.samples[self.current_processing_sample_index]
            nav_text += f" | processing {self.current_processing_sample_index + 1}/{len(self.samples)} {processing.sample_id}"
        self.sample_nav_var.set(nav_text)
        self.previous_sample_button.configure(state="normal" if self.current_sample_index > 0 else "disabled")
        self.next_sample_button.configure(state="normal" if self.current_sample_index < len(self.samples) - 1 else "disabled")

    def start_processing(self) -> None:
        if not self.samples:
            messagebox.showerror("Start inference", "Select a checkpoint and sample folder first.")
            return

        if self.is_running:
            return

        pending_fovs = [
            fov
            for session in self.samples
            for fov in session.fovs
            if fov.stage in {"Pending", "Error"}
        ]
        if not pending_fovs:
            self._update_status("All loaded samples are already complete.")
            self._update_button_states()
            return

        if self.is_paused and self.worker_pause_event is not None:
            self.worker_pause_event.clear()
            self.is_paused = False
            self.is_running = True
            self._update_status("Resuming sample queue.")
            self._update_button_states()
            return

        self.current_job_id += 1
        job_id = self.current_job_id
        self.worker_cancel_event = threading.Event()
        self.worker_pause_event = threading.Event()
        self.is_running = True
        self.is_paused = False
        self.current_processing_index = None

        self.worker_thread = threading.Thread(
            target=self._run_inference_worker,
            args=(job_id, list(self.samples), self.worker_cancel_event, self.worker_pause_event),
            daemon=True,
        )
        self.worker_thread.start()
        self._update_status(f"Loading model for {len(self.samples)} sample(s)...")
        self._update_button_states()

    def pause_processing(self) -> None:
        if not self.is_running or self.worker_pause_event is None:
            return
        self.worker_pause_event.set()
        self.is_running = False
        self.is_paused = True
        if self.samples:
            self._update_status("Paused sample queue.")
        self._update_button_states()

    def reset_processing(self) -> None:
        self._cancel_current_job()
        if self.samples:
            for session in self.samples:
                for fov in session.fovs:
                    fov.stage = "Pending"
                    fov.result = None
                    fov.error = ""
                    fov.result_version += 1
                    fov.inference_seconds = None
            self.current_processing_index = None
            self.current_processing_sample_index = None
            self._update_status("Reset loaded samples.")
            self._update_sample_nav()
        self.refresh_ui()

    def _cancel_current_job(self) -> None:
        if self.worker_cancel_event is not None:
            self.worker_cancel_event.set()
        if self.worker_pause_event is not None:
            self.worker_pause_event.clear()
        self.is_running = False
        self.is_paused = False
        self.worker_thread = None
        self.worker_cancel_event = None
        self.worker_pause_event = None
        self.current_processing_index = None
        self.current_processing_sample_index = None
        self._update_sample_nav()
        self._update_button_states()

    def _run_inference_worker(
        self,
        job_id: int,
        sessions: list[SampleSession],
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> None:
        try:
            if not sessions:
                self.job_queue.put((job_id, "done", None))
                return

            runtime_session = sessions[0]
            self.job_queue.put((job_id, "status", f"Loading model from {runtime_session.checkpoint_path.name}..."))
            runtime = build_runtime(runtime_session)
            self.job_queue.put((job_id, "runtime", runtime))

            for sample_index, session in enumerate(sessions):
                self.job_queue.put((job_id, "sample_start", sample_index))
                total = len(session.fovs)
                for index, fov in enumerate(session.fovs):
                    if cancel_event.is_set():
                        self.job_queue.put((job_id, "cancelled", None))
                        return

                    if fov.result is not None:
                        continue

                    while pause_event.is_set():
                        if cancel_event.is_set():
                            self.job_queue.put((job_id, "cancelled", None))
                            return
                        time.sleep(0.1)

                    self.job_queue.put(
                        (
                            job_id,
                            "processing",
                            (sample_index, index, f"{session.sample_id}: processing {index + 1}/{total}: {fov.image_name}"),
                        )
                    )
                    started_at = time.perf_counter()
                    try:
                        result = run_inference_on_image(fov.image_path, runtime)
                        elapsed_seconds = time.perf_counter() - started_at
                        self.job_queue.put((job_id, "result", (sample_index, index, result, elapsed_seconds)))
                    except Exception as exc:
                        elapsed_seconds = time.perf_counter() - started_at
                        self.job_queue.put((job_id, "image_error", (sample_index, index, str(exc), elapsed_seconds)))
                self.job_queue.put((job_id, "sample_done", sample_index))

            self.job_queue.put((job_id, "done", None))
        except Exception as exc:
            tb = traceback.format_exc()
            self.job_queue.put((job_id, "fatal_error", (str(exc), tb)))

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                job_id, event_type, payload = self.job_queue.get_nowait()
                if job_id != self.current_job_id:
                    continue

                if event_type == "status":
                    self._update_status(str(payload))
                elif event_type == "runtime":
                    runtime = payload
                    if isinstance(runtime, InferenceRuntime):
                        self.model_class_var.set(runtime.model_class)
                        self.resolution_var.set(str(runtime.model_resolution) if runtime.model_resolution is not None else "Unknown")
                        self.class_names_var.set(", ".join(runtime.class_names))
                        for session in self.samples:
                            session.class_names = list(runtime.class_names)
                            session.class_score_thresholds = dict(runtime.class_score_thresholds)
                elif event_type == "sample_start":
                    self.current_processing_sample_index = int(payload)
                    self._update_sample_nav()
                elif event_type == "processing":
                    sample_index, index, status_text = payload
                    self.current_processing_index = int(index)
                    if 0 <= sample_index < len(self.samples):
                        self.current_processing_sample_index = int(sample_index)
                        session = self.samples[sample_index]
                        if 0 <= index < len(session.fovs):
                            session.fovs[index].stage = "Processing"
                    self._update_status(status_text)
                    self._update_sample_nav()
                    if sample_index == self.current_sample_index:
                        self.refresh_inference_progress_ui()
                elif event_type == "result":
                    sample_index, index, result, elapsed_seconds = payload
                    if 0 <= sample_index < len(self.samples):
                        session = self.samples[sample_index]
                        if 0 <= index < len(session.fovs):
                            fov = session.fovs[index]
                            fov.stage = "Complete"
                            fov.result = result
                            fov.error = ""
                            fov.result_version += 1
                            fov.inference_seconds = float(elapsed_seconds)
                            self.current_processing_index = None
                            if sample_index == self.current_sample_index and self.selected_fov is None:
                                self.selected_fov = fov
                    if sample_index == self.current_sample_index:
                        self.refresh_inference_progress_ui()
                elif event_type == "image_error":
                    sample_index, index, error_message, elapsed_seconds = payload
                    if 0 <= sample_index < len(self.samples):
                        session = self.samples[sample_index]
                        if 0 <= index < len(session.fovs):
                            fov = session.fovs[index]
                            fov.stage = "Error"
                            fov.error = error_message
                            fov.result = None
                            fov.result_version += 1
                            fov.inference_seconds = float(elapsed_seconds)
                            self.current_processing_index = None
                    self._update_status(error_message)
                    if sample_index == self.current_sample_index:
                        self.refresh_inference_progress_ui()
                elif event_type == "sample_done":
                    sample_index = int(payload)
                    if sample_index == self.current_sample_index:
                        self.refresh_ui()
                elif event_type == "done":
                    self.is_running = False
                    self.is_paused = False
                    self.current_processing_index = None
                    self.current_processing_sample_index = None
                    if self.samples:
                        self._update_status("Completed loaded sample queue.")
                    self._update_sample_nav()
                    self._update_button_states()
                    self.refresh_ui()
                elif event_type == "cancelled":
                    self.is_running = False
                    self.is_paused = False
                    self.current_processing_index = None
                    self.current_processing_sample_index = None
                    self._update_sample_nav()
                    self._update_button_states()
                elif event_type == "fatal_error":
                    self.is_running = False
                    self.is_paused = False
                    self.current_processing_index = None
                    self.current_processing_sample_index = None
                    error_message, tb = payload
                    self._update_status(error_message)
                    self._update_sample_nav()
                    self._update_button_states()
                    self.refresh_ui()
                    messagebox.showerror("Inference failed", f"{error_message}\n\n{tb}")
        except queue.Empty:
            pass
        finally:
            self._update_button_states()
            self.root.after(100, self._poll_worker_queue)

    def _update_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _update_button_states(self) -> None:
        has_sample = bool(self.samples)
        self.load_button.configure(state="normal")
        self.start_button.configure(state="disabled")
        self.pause_button.configure(state="disabled")
        self.reset_button.configure(state="normal" if has_sample else "disabled")
        self._update_sample_nav()

        if not has_sample:
            return

        if self.is_running:
            self.start_button.configure(text="Running", state="disabled")
            self.pause_button.configure(state="normal")
            return

        if self.is_paused:
            self.start_button.configure(text="Resume", state="normal")
            return

        pending_count = sum(1 for session in self.samples for fov in session.fovs if fov.result is None)
        self.start_button.configure(text="Start", state="normal" if pending_count > 0 else "disabled")

    def refresh_ui(
        self,
        draw_slide: bool = True,
        draw_heatmaps: bool = True,
        update_detail: bool = True,
    ) -> None:
        self._update_summary()
        self.update_ranking()
        if update_detail:
            self._update_selected_detail()
        if draw_slide:
            self.draw_slide()
        if draw_heatmaps:
            self.draw_heatmaps()

    def refresh_inference_progress_ui(self) -> None:
        self._update_summary()
        if self.live_slide_updates.get():
            self.draw_slide()
        if self.live_heatmap_updates.get():
            self.draw_heatmaps()

    def _update_summary(self) -> None:
        if self.sample is None:
            self.present_var.set("0")
            self.completed_var.set("0 / 0")
            self.pending_var.set("0")
            self.qualified_var.set("0")
            self.partial_var.set("0")
            self.not_qualified_var.set("0")
            self.leucocyte_total_var.set("0")
            self.squamous_total_var.set("0")
            self.inference_time_var.set("-")
            self.inference_progress.configure(maximum=1, value=0)
            self.overall_badge.configure(text="Awaiting QA", bg=QA_COLORS["Pending"])
            return

        total = len(self.sample.fovs)
        completed = [fov for fov in self.sample.fovs if fov.result is not None]
        counts = Counter(fov.result.predicted_label for fov in completed if fov.result is not None)

        self.present_var.set(str(total))
        self.completed_var.set(f"{len(completed)} / {total}")
        self.pending_var.set(str(total - len(completed)))
        self.qualified_var.set(str(counts.get("Qualified", 0)))
        self.partial_var.set(str(counts.get("Partially qualified", 0)))
        self.not_qualified_var.set(str(counts.get("Not qualified", 0)))
        self.leucocyte_total_var.set(str(sum(fov.result.n_leucocyte for fov in completed if fov.result is not None)))
        self.squamous_total_var.set(str(sum(fov.result.n_squamous_epithelial_cell for fov in completed if fov.result is not None)))
        timed_fovs = [fov.inference_seconds for fov in self.sample.fovs if fov.inference_seconds is not None]
        if timed_fovs:
            average_seconds = sum(timed_fovs) / len(timed_fovs)
            self.inference_time_var.set(f"{average_seconds:.2f} s")
        else:
            self.inference_time_var.set("-")

        self.inference_progress.configure(maximum=max(total, 1), value=len(completed))

        overall = overall_qa_label(completed, self.overall_rule.get())
        badge_color = QA_COLORS.get(overall, QA_COLORS["Pending"])
        badge_text = overall if overall != "Pending" else "Awaiting QA"
        self.overall_badge.configure(text=badge_text, bg=badge_color)

    def _ranking_score(self, fov: FOVRecord, max_leucocytes: int, max_epithelial: int) -> float:
        if fov.result is None:
            return float("-inf")

        leucocyte_component = fov.result.n_leucocyte / max(max_leucocytes, 1)
        epithelial_component = 1.0 - (fov.result.n_squamous_epithelial_cell / max(max_epithelial, 1))
        return leucocyte_component + epithelial_component

    def _ranked_fovs(self) -> list[tuple[int, FOVRecord]]:
        if self.sample is None:
            return []

        completed_fovs = [fov for fov in self.sample.fovs if fov.result is not None]
        if not completed_fovs:
            return []

        max_leucocytes = max((fov.result.n_leucocyte for fov in completed_fovs if fov.result is not None), default=0)
        max_epithelial = max(
            (fov.result.n_squamous_epithelial_cell for fov in completed_fovs if fov.result is not None),
            default=0,
        )

        ranked_rows = [
            (
                self._ranking_score(fov, max_leucocytes, max_epithelial),
                fov,
            )
            for fov in completed_fovs
        ]
        ranked_rows.sort(
            key=lambda item: (
                SEVERITY.get(item[1].result.predicted_label, 99) if item[1].result is not None else 99,
                -item[0],
                item[1].result.n_squamous_epithelial_cell if item[1].result is not None else 10**9,
                -(item[1].result.n_leucocyte if item[1].result is not None else -1),
                item[1].ingest_index,
            )
        )
        return [(rank, fov) for rank, (_score, fov) in enumerate(ranked_rows, start=1)]

    def update_ranking(self) -> None:
        self.draw_ranking()

    def _select_ranking_item_for_fov(self, fov: FOVRecord) -> None:
        self.draw_ranking()

    def draw_ranking(self) -> None:
        self.ranking_canvas.delete("all")
        width = max(self.ranking_canvas.winfo_width(), 100)
        height = max(self.ranking_canvas.winfo_height(), 80)
        self.ranking_canvas.configure(scrollregion=(0, 0, width, height))
        self.ranking_canvas.create_rectangle(0, 0, width, height, fill=SLIDE_BG, outline="")
        self.ranking_geometry = (0.0, 0.0, 0.0)

        if self.sample is None:
            self.ranking_canvas.create_text(width / 2, height / 2, text="No sample", fill="#94a3b8", font=("Segoe UI", 10))
            return

        grid_width = max(self.sample.grid_width, 1)
        grid_height = max(self.sample.grid_height, 1)
        padding = 10
        label_band = 20
        usable_width = max(width - (padding * 2) - label_band, 60)
        usable_height = max(height - (padding * 2) - label_band, 50)
        cell_size = max(12, int(min(usable_width / grid_width, usable_height / grid_height)))
        map_width = grid_width * cell_size
        map_height = grid_height * cell_size
        origin_x = (width - map_width) / 2 + (label_band / 2)
        origin_y = (height - map_height) / 2 + (label_band / 2)
        self.ranking_geometry = (origin_x, origin_y, cell_size)

        rank_by_fov = {fov.ingest_index: rank for rank, fov in self._ranked_fovs()}
        by_position = self.sample.by_position
        for row in range(grid_height):
            for col in range(grid_width):
                x0 = origin_x + (col * cell_size)
                y0 = origin_y + (row * cell_size)
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                fov = by_position.get((col, row))
                fill = "#0b1220"
                text = ""
                text_fill = "#cbd5e1"

                if fov is not None:
                    if fov.result is not None:
                        fill = QA_COLORS.get(fov.result.predicted_label, QA_COLORS["Pending"])
                        text = str(rank_by_fov.get(fov.ingest_index, ""))
                        text_fill = "white"
                    elif fov.stage == "Processing":
                        fill = QA_COLORS["Processing"]
                    elif fov.stage == "Error":
                        fill = QA_COLORS["Error"]
                    else:
                        fill = QA_COLORS["Pending"]

                self.ranking_canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="")
                if text and cell_size >= 16:
                    font_size = 8 if cell_size < 24 else 10
                    self.ranking_canvas.create_text(
                        x0 + cell_size / 2,
                        y0 + cell_size / 2,
                        text=text,
                        fill=text_fill,
                        font=("Segoe UI", font_size, "bold"),
                    )
                if self.selected_fov is fov:
                    self.ranking_canvas.create_rectangle(x0, y0, x1, y1, outline="#f8fafc", width=2)

    def _fov_from_ranking_event(self, event: tk.Event[tk.Canvas]) -> FOVRecord | None:
        if self.sample is None:
            return None
        origin_x, origin_y, cell_size = self.ranking_geometry
        if cell_size <= 0:
            return None

        col = int((event.x - origin_x) // cell_size)
        row = int((event.y - origin_y) // cell_size)
        if col < 0 or row < 0 or col >= self.sample.grid_width or row >= self.sample.grid_height:
            return None
        return self.sample.by_position.get((col, row))

    def _on_ranking_click(self, event: tk.Event[tk.Canvas]) -> None:
        fov = self._fov_from_ranking_event(event)
        if fov is None:
            return

        self.selected_fov = fov
        self._update_selected_detail()
        self.draw_ranking()
        self.draw_slide()

    def _on_ranking_double_click(self, event: tk.Event[tk.Canvas]) -> None:
        fov = self._fov_from_ranking_event(event)
        if fov is None:
            return

        self.selected_fov = fov
        self._update_selected_detail()
        self._select_ranking_item_for_fov(fov)
        self.draw_slide()
        self.open_fov_inspector(fov)

    def _update_selected_detail(self) -> None:
        fov = self.selected_fov
        if fov is None:
            self.fov_name_var.set("-")
            self.fov_position_var.set("-")
            self.fov_stage_var.set("-")
            self.fov_qa_var.set("-")
            self.fov_leucocyte_var.set("-")
            self.fov_squamous_var.set("-")
            self.fov_predictions_var.set("-")
            self.fov_score_var.set("-")
            self.preview_photo = ImageTk.PhotoImage(make_placeholder_image((SELECTED_PREVIEW_SIZE, SELECTED_PREVIEW_SIZE), "No FOV", "#9ca3af"))
            self.preview_label.configure(image=self.preview_photo)
            return

        self.fov_name_var.set(Path(fov.image_name).stem)
        self.fov_position_var.set(f"{fov.coord_x}, {fov.coord_y}")
        self.fov_stage_var.set(fov.stage)

        if fov.result is not None:
            self.fov_qa_var.set(fov.result.predicted_label)
            self.fov_leucocyte_var.set(str(fov.result.n_leucocyte))
            self.fov_squamous_var.set(str(fov.result.n_squamous_epithelial_cell))
            self.fov_predictions_var.set(str(fov.result.n_predictions_kept))
            self.fov_score_var.set(f"{fov.result.total_quality_score:.1f}")
        elif fov.stage == "Error":
            self.fov_qa_var.set("Inference error")
            self.fov_leucocyte_var.set("-")
            self.fov_squamous_var.set("-")
            self.fov_predictions_var.set("-")
            self.fov_score_var.set("-")
        else:
            self.fov_qa_var.set("Awaiting QA")
            self.fov_leucocyte_var.set("-")
            self.fov_squamous_var.set("-")
            self.fov_predictions_var.set("-")
            self.fov_score_var.set("-")

        preview_image = self._preview_photo_for_fov(fov, SELECTED_PREVIEW_SIZE, SELECTED_PREVIEW_SIZE)
        self.preview_photo = preview_image
        self.preview_label.configure(image=self.preview_photo)

    def _preview_photo_for_fov(self, fov: FOVRecord, width: int, height: int) -> ImageTk.PhotoImage:
        if fov.result is not None:
            try:
                overlay = render_result_overlay(fov.image_path, fov.result, self.sample.class_names if self.sample else fov_eval.DEFAULT_CLASS_NAMES)
                fitted = ImageOps.fit(overlay, (width, height), Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(fitted)
            except Exception:
                pass

        try:
            with Image.open(fov.image_path) as raw_img:
                image = ImageOps.fit(raw_img.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image)
        except Exception:
            placeholder_text = "Image error" if fov.stage == "Error" else f"{fov.coord_x},{fov.coord_y}"
            return ImageTk.PhotoImage(make_placeholder_image((width, height), placeholder_text, self._fov_border_color(fov)))

    def _tile_photo_for_fov(self, fov: FOVRecord, width: int, height: int) -> ImageTk.PhotoImage:
        cache_key = (str(fov.image_path), width, height)
        if cache_key in self.photo_cache:
            return self.photo_cache[cache_key]

        try:
            with Image.open(fov.image_path) as raw_img:
                image = ImageOps.fit(raw_img.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
        except Exception:
            image = make_placeholder_image((width, height), f"{fov.coord_x},{fov.coord_y}", self._fov_border_color(fov))

        photo = ImageTk.PhotoImage(image)
        self.photo_cache[cache_key] = photo
        return photo

    def _fov_border_color(self, fov: FOVRecord) -> str:
        if fov.result is not None:
            return QA_COLORS.get(fov.result.predicted_label, QA_COLORS["Pending"])
        if fov.stage == "Processing":
            return QA_COLORS["Processing"]
        if fov.stage == "Error":
            return QA_COLORS["Error"]
        return QA_COLORS["Pending"]

    def _class_index_for_heatmap(self, class_type: str) -> int | None:
        if self.sample is None:
            return None

        for index, class_name in enumerate(self.sample.class_names):
            normalized = re.sub(r"[^a-z0-9]+", "", class_name.lower())
            if class_type == "leucocyte" and any(token in normalized for token in ("leucocyte", "leukocyte", "wbc")):
                return index
            if class_type == "epithelial" and any(token in normalized for token in ("squamous", "epithelial")):
                return index
        return None

    def draw_heatmaps(self) -> None:
        leucocyte_index = self._class_index_for_heatmap("leucocyte")
        epithelial_index = self._class_index_for_heatmap("epithelial")
        source_fovs = self._filtered_heatmap_fovs()
        filter_label = self.heatmap_filter.get()
        self._draw_count_heatmap(
            self.leucocyte_heatmap_canvas,
            leucocyte_index,
            source_fovs=source_fovs,
            filter_label=filter_label,
            high_color="#ef4444",
            empty_text="Leucocyte map awaits QA",
        )
        self._draw_count_heatmap(
            self.epithelial_heatmap_canvas,
            epithelial_index,
            source_fovs=source_fovs,
            filter_label=filter_label,
            high_color="#22c55e",
            empty_text="Epithelial map awaits QA",
        )

    def _filtered_heatmap_fovs(self) -> list[FOVRecord]:
        if self.sample is None:
            return []

        completed_fovs = [fov for fov in self.sample.fovs if fov.result is not None]
        selected_filter = self.heatmap_filter.get()
        if selected_filter == "All":
            return completed_fovs
        return [
            fov
            for fov in completed_fovs
            if fov.result is not None and fov.result.predicted_label == selected_filter
        ]

    def _draw_count_heatmap(
        self,
        canvas: tk.Canvas,
        class_index: int | None,
        source_fovs: list[FOVRecord],
        filter_label: str,
        high_color: str,
        empty_text: str,
    ) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 80)
        height = max(canvas.winfo_height(), 80)
        canvas.create_rectangle(0, 0, width, height, fill="#111827", outline="")

        if self.sample is None:
            canvas.configure(scrollregion=(0, 0, width, height))
            canvas.create_text(width / 2, height / 2, text=empty_text, fill="#94a3b8", font=("Segoe UI", 10))
            return

        if class_index is None:
            canvas.configure(scrollregion=(0, 0, width, height))
            canvas.create_text(width / 2, height / 2, text="Class not found in model", fill="#fca5a5", font=("Segoe UI", 10, "bold"))
            return

        grid_width = max(self.sample.grid_width, 1)
        grid_height = max(self.sample.grid_height, 1)
        margin = 12
        header_height = 28
        usable_width = max(width - margin * 2, 40)
        usable_height = max(height - margin * 2 - header_height, 40)
        cell_size = max(4, min(usable_width / grid_width, usable_height / grid_height))
        map_width = grid_width * cell_size
        map_height = grid_height * cell_size
        origin_x = (width - map_width) / 2
        origin_y = margin + header_height + max(0, (usable_height - map_height) / 2)
        canvas.configure(scrollregion=(0, 0, width, height))

        total_objects = 0
        fov_counts: dict[tuple[int, int], int] = {}
        selected_positions = {(fov.coord_x, fov.coord_y) for fov in source_fovs}
        for fov in source_fovs:
            count = sum(1 for cls_idx in fov.result.pred_cls if int(cls_idx) == int(class_index))
            fov_counts[(fov.coord_x, fov.coord_y)] = count
            total_objects += count

        max_count = max(fov_counts.values(), default=0)
        zero_color = "#172033"
        pending_color = "#1f2937"
        missing_color = "#0b1220"
        by_position = self.sample.by_position
        for row in range(grid_height):
            for col in range(grid_width):
                x0 = origin_x + col * cell_size
                y0 = origin_y + row * cell_size
                fov = by_position.get((col, row))
                fill = missing_color
                label = ""
                text_fill = "#94a3b8"

                if fov is not None:
                    if fov.result is None:
                        fill = pending_color if filter_label == "All" else missing_color
                    elif (col, row) in selected_positions:
                        count = fov_counts.get((col, row), 0)
                        label = str(count)
                        if count > 0 and max_count > 0:
                            fill = _blend_hex_color(zero_color, high_color, count / max_count)
                            text_fill = "#f8fafc"
                        else:
                            fill = zero_color

                canvas.create_rectangle(x0, y0, x0 + cell_size, y0 + cell_size, fill=fill, outline="")
                if label and cell_size >= 22:
                    canvas.create_text(
                        x0 + cell_size / 2,
                        y0 + cell_size / 2,
                        text=label,
                        fill=text_fill,
                        font=("Segoe UI", 8, "bold"),
                    )

        if filter_label == "All":
            completed_count = sum(1 for fov in self.sample.fovs if fov.result is not None)
            processed_label = f"All: {completed_count} / {len(self.sample.fovs)} FOVs"
        else:
            processed_label = f"{filter_label}: {len(source_fovs)} FOVs"
        count_label = f"{total_objects} objects | max/FOV {max_count}"
        canvas.create_text(margin, 8, text=processed_label, fill="#dbeafe", font=("Segoe UI", 9), anchor="nw")
        canvas.create_text(width - margin, 8, text=count_label, fill="#dbeafe", font=("Segoe UI", 9, "bold"), anchor="ne")

        if filter_label != "All" and not source_fovs:
            message = empty_text if filter_label == "All" else f"No {filter_label.lower()} FOVs yet"
            canvas.create_text(width / 2, height / 2, text=message, fill="#94a3b8", font=("Segoe UI", 10))

    def _inspection_image_for_fov(self, fov: FOVRecord) -> Image.Image:
        if fov.result is not None:
            return render_result_overlay(
                fov.image_path,
                fov.result,
                self.sample.class_names if self.sample else fov_eval.DEFAULT_CLASS_NAMES,
            )

        with Image.open(fov.image_path) as raw_img:
            return raw_img.convert("RGB")

    def open_fov_inspector(self, fov: FOVRecord) -> None:
        try:
            with Image.open(fov.image_path) as raw_img:
                raw_image = raw_img.convert("RGB")
            prediction_image = (
                render_result_overlay(
                    fov.image_path,
                    fov.result,
                    self.sample.class_names if self.sample else fov_eval.DEFAULT_CLASS_NAMES,
                )
                if fov.result is not None
                else raw_image.copy()
            )
        except Exception as exc:
            messagebox.showerror("Open FOV", f"Could not open image:\n{exc}")
            return

        window = tk.Toplevel(self.root)
        window.title(f"{Path(fov.image_name).stem} - FOV inspection")
        window.geometry("1180x860")
        window.minsize(700, 500)
        window.configure(bg=APP_BG)
        window.rowconfigure(1, weight=1)
        window.columnconfigure(0, weight=1)

        toolbar = ttk.Frame(window, style="App.TFrame", padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(0, weight=1)

        if fov.result is not None:
            info_text = (
                f"{Path(fov.image_name).stem} | Position {fov.coord_x}, {fov.coord_y} | "
                f"{fov.result.predicted_label} | "
                f"Leucocytes {fov.result.n_leucocyte} | "
                f"Squamous cells {fov.result.n_squamous_epithelial_cell} | "
                f"Objects {fov.result.n_predictions_kept}"
            )
        else:
            info_text = f"{Path(fov.image_name).stem} | Position {fov.coord_x}, {fov.coord_y} | Raw image, QA not complete"

        tk.Label(
            toolbar,
            text=info_text,
            bg=APP_BG,
            fg=TEXT_PRIMARY,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 10))

        viewer = ttk.Frame(window, style="App.TFrame")
        viewer.grid(row=1, column=0, sticky="nsew")
        viewer.rowconfigure(1, weight=1)
        viewer.columnconfigure(0, weight=1)
        viewer.columnconfigure(1, weight=1)

        prediction_title = "Predictions" if fov.result is not None else "Predictions pending"
        tk.Label(
            viewer,
            text=prediction_title,
            bg=APP_BG,
            fg=TEXT_PRIMARY,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            padx=8,
            pady=6,
        ).grid(row=0, column=0, sticky="ew")
        tk.Label(
            viewer,
            text="Raw image",
            bg=APP_BG,
            fg=TEXT_PRIMARY,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            padx=8,
            pady=6,
        ).grid(row=0, column=1, sticky="ew")

        prediction_canvas = tk.Canvas(viewer, bg="#0f172a", highlightthickness=0)
        prediction_canvas.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        raw_canvas = tk.Canvas(viewer, bg="#0f172a", highlightthickness=0)
        raw_canvas.grid(row=1, column=1, sticky="nsew", padx=(4, 0))

        def shared_yview(*args: Any) -> None:
            prediction_canvas.yview(*args)
            raw_canvas.yview(*args)

        def shared_xview(*args: Any) -> None:
            prediction_canvas.xview(*args)
            raw_canvas.xview(*args)

        v_scroll = ttk.Scrollbar(viewer, orient="vertical", command=shared_yview)
        v_scroll.grid(row=1, column=2, sticky="ns")
        h_scroll = ttk.Scrollbar(viewer, orient="horizontal", command=shared_xview)
        h_scroll.grid(row=2, column=0, columnspan=2, sticky="ew")
        prediction_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        state: dict[str, Any] = {
            "scale": 1.0,
            "prediction_photo": None,
            "raw_photo": None,
            "prediction_image_id": None,
            "raw_image_id": None,
        }

        def render_at_scale(scale: float) -> None:
            scale = max(0.05, min(8.0, float(scale)))
            x_fraction = prediction_canvas.xview()[0] if prediction_canvas.xview() else 0.0
            y_fraction = prediction_canvas.yview()[0] if prediction_canvas.yview() else 0.0
            state["scale"] = scale
            width = max(1, int(raw_image.width * scale))
            height = max(1, int(raw_image.height * scale))

            prediction_photo = ImageTk.PhotoImage(prediction_image.resize((width, height), Image.Resampling.LANCZOS))
            raw_photo = ImageTk.PhotoImage(raw_image.resize((width, height), Image.Resampling.LANCZOS))
            state["prediction_photo"] = prediction_photo
            state["raw_photo"] = raw_photo

            if state["prediction_image_id"] is None:
                state["prediction_image_id"] = prediction_canvas.create_image(0, 0, image=prediction_photo, anchor="nw")
            else:
                prediction_canvas.itemconfigure(state["prediction_image_id"], image=prediction_photo)

            if state["raw_image_id"] is None:
                state["raw_image_id"] = raw_canvas.create_image(0, 0, image=raw_photo, anchor="nw")
            else:
                raw_canvas.itemconfigure(state["raw_image_id"], image=raw_photo)

            for canvas in (prediction_canvas, raw_canvas):
                canvas.configure(scrollregion=(0, 0, width, height))
                canvas.xview_moveto(x_fraction)
                canvas.yview_moveto(y_fraction)

        def fit_to_window() -> None:
            window.update_idletasks()
            available_width = max(200, min(prediction_canvas.winfo_width(), raw_canvas.winfo_width()) - 12)
            available_height = max(200, min(prediction_canvas.winfo_height(), raw_canvas.winfo_height()) - 12)
            scale = min(
                available_width / max(1, raw_image.width),
                available_height / max(1, raw_image.height),
                1.0,
            )
            render_at_scale(scale)

        def zoom(factor: float) -> None:
            render_at_scale(state["scale"] * factor)

        ttk.Button(toolbar, text="Zoom in", command=lambda: zoom(1.25)).grid(row=0, column=1, padx=3)
        ttk.Button(toolbar, text="Zoom out", command=lambda: zoom(0.80)).grid(row=0, column=2, padx=3)
        ttk.Button(toolbar, text="100%", command=lambda: render_at_scale(1.0)).grid(row=0, column=3, padx=3)
        ttk.Button(toolbar, text="Fit", command=fit_to_window).grid(row=0, column=4, padx=(3, 0))

        def on_control_wheel(event: tk.Event[tk.Canvas]) -> str:
            zoom(1.12 if event.delta > 0 else 0.89)
            return "break"

        def on_mouse_wheel(event: tk.Event[tk.Canvas]) -> str:
            direction = -1 if event.delta > 0 else 1
            shared_yview("scroll", direction, "units")
            return "break"

        def on_shift_mouse_wheel(event: tk.Event[tk.Canvas]) -> str:
            direction = -1 if event.delta > 0 else 1
            shared_xview("scroll", direction, "units")
            return "break"

        for canvas in (prediction_canvas, raw_canvas):
            canvas.bind("<Control-MouseWheel>", on_control_wheel)
            canvas.bind("<MouseWheel>", on_mouse_wheel)
            canvas.bind("<Shift-MouseWheel>", on_shift_mouse_wheel)
        window.after(100, fit_to_window)

    def draw_slide(self) -> None:
        self.slide_canvas.delete("all")
        self._canvas_images.clear()
        width = max(self.slide_canvas.winfo_width(), 100)
        height = max(self.slide_canvas.winfo_height(), 100)

        if self.sample is None:
            self.slide_geometry = (0.0, 0.0, 0.0)
            self.slide_canvas.configure(scrollregion=(0, 0, width, height))
            return

        grid_width = max(self.sample.grid_width, 1)
        grid_height = max(self.sample.grid_height, 1)
        padding = 36
        label_band = 24
        usable_width = max(width - (padding * 2) - label_band, 80)
        usable_height = max(height - (padding * 2) - label_band, 80)
        fit_cell_size = min(usable_width / grid_width, usable_height / grid_height)
        cell_size = max(16, int(fit_cell_size * self.slide_zoom))

        slide_width = grid_width * cell_size
        slide_height = grid_height * cell_size
        content_width = slide_width + (padding * 2) + label_band
        content_height = slide_height + (padding * 2) + label_band
        scroll_width = max(width, content_width)
        scroll_height = max(height, content_height)
        origin_x = padding + label_band
        origin_y = padding + label_band
        if content_width <= width:
            origin_x = (width - slide_width) / 2 + (label_band / 2)
        if content_height <= height:
            origin_y = (height - slide_height) / 2 + (label_band / 2)
        self.slide_geometry = (origin_x, origin_y, cell_size)
        self.slide_canvas.configure(scrollregion=(0, 0, scroll_width, scroll_height))

        self.slide_canvas.create_rectangle(
            origin_x - 12,
            origin_y - 12,
            origin_x + slide_width + 12,
            origin_y + slide_height + 12,
            fill="#1f2937",
            outline="#374151",
            width=2,
        )

        for col in range(grid_width):
            x = origin_x + (col * cell_size) + (cell_size / 2)
            self.slide_canvas.create_text(x, origin_y - 18, text=str(col), fill="#cbd5e1", font=("Segoe UI", 9))

        for row in range(grid_height):
            y = origin_y + (row * cell_size) + (cell_size / 2)
            self.slide_canvas.create_text(origin_x - 18, y, text=str(row), fill="#cbd5e1", font=("Segoe UI", 9))

        by_position = self.sample.by_position
        for row in range(grid_height):
            for col in range(grid_width):
                x0 = origin_x + (col * cell_size)
                y0 = origin_y + (row * cell_size)
                x1 = x0 + cell_size
                y1 = y0 + cell_size
                fov = by_position.get((col, row))

                self.slide_canvas.create_rectangle(x0, y0, x1, y1, fill="#1f2937", outline="#334155", width=1)
                if fov is None:
                    continue

                tile_size = max(8, cell_size - 8)
                photo = self._tile_photo_for_fov(fov, tile_size, tile_size)
                self._canvas_images.append(photo)
                self.slide_canvas.create_image(x0 + 4, y0 + 4, anchor="nw", image=photo)

                self.slide_canvas.create_rectangle(
                    x0 + 2,
                    y0 + 2,
                    x1 - 2,
                    y1 - 2,
                    outline=self._fov_border_color(fov),
                    width=3,
                )

                if self.selected_fov is fov:
                    self.slide_canvas.create_rectangle(
                        x0,
                        y0,
                        x1,
                        y1,
                        outline="#f8fafc",
                        width=2,
                    )

    def _fov_from_canvas_event(self, event: tk.Event[tk.Canvas]) -> FOVRecord | None:
        if self.sample is None:
            return None
        origin_x, origin_y, cell_size = self.slide_geometry
        if cell_size <= 0:
            return None

        canvas_x = self.slide_canvas.canvasx(event.x)
        canvas_y = self.slide_canvas.canvasy(event.y)
        col = int((canvas_x - origin_x) // cell_size)
        row = int((canvas_y - origin_y) // cell_size)
        if col < 0 or row < 0 or col >= self.sample.grid_width or row >= self.sample.grid_height:
            return None

        return self.sample.by_position.get((col, row))

    def _on_canvas_click(self, event: tk.Event[tk.Canvas]) -> None:
        fov = self._fov_from_canvas_event(event)
        if fov is None:
            return

        self.selected_fov = fov
        self._update_selected_detail()
        self.draw_slide()

    def _on_canvas_double_click(self, event: tk.Event[tk.Canvas]) -> None:
        fov = self._fov_from_canvas_event(event)
        if fov is None:
            return

        self.selected_fov = fov
        self._update_selected_detail()
        self.draw_slide()
        self.open_fov_inspector(fov)

    def _on_close(self) -> None:
        self._cancel_current_job()
        self.root.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive UI for live full-FOV quality assessment.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.exists() else "", help="Initial checkpoint path.")
    parser.add_argument("--sample-folder", default="", help="Initial sample folder.")
    parser.add_argument("--auto-load", action="store_true", help="Auto-load the checkpoint and sample folder on startup.")
    parser.add_argument("--check", action="store_true", help="Validate checkpoint and sample discovery without launching the UI.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else None
    sample_folder = Path(args.sample_folder).expanduser() if args.sample_folder else None

    if args.check:
        check_dependencies()
        if checkpoint and sample_folder:
            session = build_sample_session(checkpoint, sample_folder)
            print(f"Sample: {session.sample_id}")
            print(f"Folder: {session.sample_dir}")
            print(f"Checkpoint: {session.checkpoint_path}")
            print(f"Model class: {session.model_class}")
            print(f"Resolution: {session.model_resolution}")
            print(f"Classes: {', '.join(session.class_names)}")
            print(f"FOV count: {len(session.fovs)}")
            print(f"Grid: {session.grid_width} x {session.grid_height}")
        else:
            print("UI dependencies are available.")
        return

    root = tk.Tk()
    QAWorkflowUI(
        root,
        initial_checkpoint=checkpoint,
        initial_sample_folder=sample_folder,
        auto_load=bool(args.auto_load),
    )
    root.mainloop()


if __name__ == "__main__":
    main()
