from __future__ import annotations

import csv
import importlib.util
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _csv_tokens(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _csv_float(raw: str) -> list[float]:
    vals = [float(x) for x in _csv_tokens(raw)]
    if not vals:
        raise ValueError("Expected non-empty float CSV")
    return vals


def _csv_int(raw: str) -> list[int]:
    vals = [int(x) for x in _csv_tokens(raw)]
    if not vals:
        raise ValueError("Expected non-empty int CSV")
    return vals


def _parse_bool(raw: str) -> bool:
    x = str(raw).strip().lower()
    if x in {"1", "true", "yes", "y", "on"}:
        return True
    if x in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected bool text, got: {raw!r}")


def _round(v: float | None) -> float | None:
    return None if v is None else float(round(float(v), 6))


def _auto_latest_dataset(dataset_root: Path, token: str) -> str:
    cands = sorted([p for p in dataset_root.glob(f"*{token}*") if p.is_dir()])
    return str(cands[-1]) if cands else ""


def _ensure_dataset_envs() -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_root = Path(os.getenv("STAT_DATASETS_ROOT", str(script_dir / "Stat_Dataset"))).expanduser().resolve()
    epi_existing = os.getenv("DATASET_EPI", "").strip()
    leu_existing = os.getenv("DATASET_LEUCO", "").strip()
    if not epi_existing:
        epi = _auto_latest_dataset(dataset_root, "SquamousEpithelialCell_OVR")
        if epi:
            os.environ["DATASET_EPI"] = epi
    if not leu_existing:
        leu = _auto_latest_dataset(dataset_root, "Leucocyte_OVR")
        if leu:
            os.environ["DATASET_LEUCO"] = leu

    # Base trainer import expects both envs; if one class dataset is unavailable,
    # mirror the available one so the module can still initialize.
    epi_now = os.getenv("DATASET_EPI", "").strip()
    leu_now = os.getenv("DATASET_LEUCO", "").strip()
    if epi_now and not leu_now:
        os.environ["DATASET_LEUCO"] = epi_now
    if leu_now and not epi_now:
        os.environ["DATASET_EPI"] = leu_now


_ensure_dataset_envs()

TARGET_KEY = os.getenv("RFDETR_SS_TARGET", "epi").strip().lower()
INIT_MODE = os.getenv("RFDETR_SS_INIT_MODE", "scratch").strip().lower()
if TARGET_KEY not in {"epi", "leu"}:
    raise ValueError("RFDETR_SS_TARGET must be epi or leu")
if INIT_MODE not in {"default", "scratch", "ssl"}:
    raise ValueError("RFDETR_SS_INIT_MODE must be default, scratch, or ssl")

TRAIN_FRACTIONS = _csv_float(os.getenv("RFDETR_SS_TRAIN_FRACTIONS", "0.25,0.10,0.05"))
SEEDS = _csv_int(os.getenv("RFDETR_SS_SEEDS", os.getenv("RFDETR_SEEDS", "42,43,44")))
FRACTION_SEEDS = _csv_int(os.getenv("RFDETR_SS_FRACTION_SEEDS", str(SEEDS[0])))
if len(FRACTION_SEEDS) == 1 and len(SEEDS) > 1:
    FRACTION_SEEDS = [FRACTION_SEEDS[0] for _ in SEEDS]
if len(FRACTION_SEEDS) not in {1, len(SEEDS)}:
    raise ValueError("RFDETR_SS_FRACTION_SEEDS must have length 1 or len(RFDETR_SS_SEEDS)")

MODEL_CLS_NAME = os.getenv("RFDETR_SS_MODEL_CLS", os.getenv("RFDETR_MODEL_CLS", "RFDETRLarge")).strip()

RFDETR_SS_EPI_EPOCHS = int(os.getenv("RFDETR_SS_EPI_EPOCHS", os.getenv("RFDETR_EPI_EPOCHS", "50")))
RFDETR_SS_LEU_EPOCHS = int(os.getenv("RFDETR_SS_LEU_EPOCHS", os.getenv("RFDETR_LEU_EPOCHS", "70")))
RFDETR_SS_EPI_LR = float(os.getenv("RFDETR_SS_EPI_LR", os.getenv("RFDETR_EPI_LR", "5e-5")))
RFDETR_SS_LEU_LR = float(os.getenv("RFDETR_SS_LEU_LR", os.getenv("RFDETR_LEU_LR", "8e-5")))

DEFAULT_BATCH = 4 if os.getenv("RFDETR_USE_PATCH_224", "0") in {"0", "false", "False"} else 16
DEFAULT_QUERIES = 120 if os.getenv("RFDETR_USE_PATCH_224", "0") in {"0", "false", "False"} else 200

RFDETR_SS_BATCH = int(os.getenv("RFDETR_SS_BATCH", os.getenv("RFDETR_EPI_BATCH", str(DEFAULT_BATCH))))
RFDETR_SS_GRAD_ACCUM = int(os.getenv("RFDETR_SS_GRAD_ACCUM_STEPS", os.getenv("RFDETR_GRAD_ACCUM_STEPS", "1")))
RFDETR_SS_WEIGHT_DECAY = float(os.getenv("RFDETR_SS_WEIGHT_DECAY", os.getenv("RFDETR_WEIGHT_DECAY", "7e-4")))
RFDETR_SS_DROPOUT = float(os.getenv("RFDETR_SS_DROPOUT", os.getenv("RFDETR_DROPOUT", "0.15")))
RFDETR_SS_NUM_QUERIES = int(os.getenv("RFDETR_SS_NUM_QUERIES", os.getenv("RFDETR_EPI_NUM_QUERIES", str(DEFAULT_QUERIES))))
RFDETR_SS_AUG_COPIES = int(os.getenv("RFDETR_SS_AUG_COPIES", os.getenv("RFDETR_AUG_COPIES", "0")))
RFDETR_SS_EARLY_STOPPING = _parse_bool(os.getenv("RFDETR_SS_EARLY_STOPPING", os.getenv("RFDETR_EARLY_STOPPING", "1")))
RFDETR_SS_EARLY_PATIENCE = int(os.getenv("RFDETR_SS_EARLY_STOPPING_PATIENCE", os.getenv("RFDETR_EARLY_STOPPING_PATIENCE", "10")))
RFDETR_SS_EARLY_MIN_DELTA = float(os.getenv("RFDETR_SS_EARLY_STOPPING_MIN_DELTA", os.getenv("RFDETR_EARLY_STOPPING_MIN_DELTA", "0.001")))
RFDETR_SS_EARLY_USE_EMA = _parse_bool(os.getenv("RFDETR_SS_EARLY_STOPPING_USE_EMA", os.getenv("RFDETR_EARLY_STOPPING_USE_EMA", "0")))

RFDETR_SS_PSEUDO_SCORE_THRESH = float(os.getenv("RFDETR_SS_PSEUDO_SCORE_THRESH", "0.70"))
RFDETR_SS_PSEUDO_MAX_DETS_PER_IMAGE = int(os.getenv("RFDETR_SS_PSEUDO_MAX_DETS_PER_IMAGE", "50"))
RFDETR_SS_UNLABELED_MAX_IMAGES = int(os.getenv("RFDETR_SS_UNLABELED_MAX_IMAGES", "0"))
RFDETR_SS_PSEUDO_IMAGES_PER_LABELED = int(os.getenv("RFDETR_SS_PSEUDO_IMAGES_PER_LABELED", "6"))
RFDETR_SS_REQUIRE_PSEUDO = _parse_bool(os.getenv("RFDETR_SS_REQUIRE_PSEUDO", "1"))
RFDETR_SS_CONTINUE_ON_ERROR = _parse_bool(os.getenv("RFDETR_SS_CONTINUE_ON_ERROR", "0"))

# Keep base import safe.
os.environ.setdefault("RFDETR_EXPERIMENT_MODE", "matrix")
os.environ.setdefault("RFDETR_INIT_MODES", INIT_MODE)
os.environ.setdefault("RFDETR_TRAIN_FRACTIONS", "1.0")
os.environ.setdefault("RFDETR_FRACTION_SEEDS", str(FRACTION_SEEDS[0]))
os.environ.setdefault("RFDETR_SEEDS", str(SEEDS[0]))

BASE_SCRIPT = Path(__file__).with_name("TrainRFDETR_SOLO_SingleClass_STATIC_Ucloud.py")
_spec = importlib.util.spec_from_file_location("rfdetr_static_base", BASE_SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load {BASE_SCRIPT}")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall  # noqa: E402
import torch  # noqa: E402


def _resolve_model_cls(name: str):
    m = {"RFDETRSmall": RFDETRSmall, "RFDETRMedium": RFDETRMedium, "RFDETRLarge": RFDETRLarge}
    if name not in m:
        raise ValueError(f"Unsupported RFDETR_SS_MODEL_CLS={name}")
    return m[name]

def _predict_one_image(model: Any, image: Image.Image, score_floor: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.array(image.convert("RGB"))
    ten = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    out = None
    for method in ("predict", "infer", "inference", "forward_inference"):
        fn = getattr(model, method, None)
        if not callable(fn):
            continue
        for inp in (image, arr, ten):
            for kwargs in ({"threshold": float(score_floor)}, {}):
                try:
                    out = fn(inp, **kwargs)
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
        fwd = getattr(model, "forward", None)
        if not callable(fwd):
            raise RuntimeError("Model has no supported inference method")
        out = fwd(ten)

    if isinstance(out, (list, tuple)) and len(out) == 1:
        out = out[0]
    if isinstance(out, (list, tuple)) and len(out) == 3:
        out = {"boxes": out[0], "scores": out[1], "labels": out[2]}

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

    if not isinstance(out, dict):
        raise RuntimeError(f"Unrecognized prediction output type: {type(out)}")

    ksets = [
        ("boxes", "scores", "labels"),
        ("pred_boxes", "scores", "labels"),
        ("bboxes", "scores", "classes"),
        ("boxes_xyxy", "scores", "labels"),
        ("detections", None, None),
    ]
    boxes = scores = labels = None
    for kb, ks, kl in ksets:
        if kb not in out or (ks is not None and ks not in out) or (kl is not None and kl not in out):
            continue
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
        raise RuntimeError(f"Unrecognized prediction keys: {list(out.keys())}")

    def _np(x: Any) -> np.ndarray:
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    boxes_np = _np(boxes).astype(np.float32)
    scores_np = _np(scores).astype(np.float32) if scores is not None else np.ones((len(boxes_np),), dtype=np.float32)
    labels_np = _np(labels).astype(np.int64) if labels is not None else np.zeros((len(boxes_np),), dtype=np.int64)
    keep = scores_np >= float(score_floor)
    return boxes_np[keep], scores_np[keep], labels_np[keep]


def _find_teacher_checkpoint(run_dir: Path) -> Path:
    for p in (
        run_dir / "checkpoint_best_total.pth",
        run_dir / "checkpoint_best_regular.pth",
        run_dir / "checkpoint_best_ema.pth",
        run_dir / "checkpoint.pth",
    ):
        if p.exists():
            return p
    found = sorted(run_dir.glob("checkpoint*.pth"))
    if found:
        return found[-1]
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def _collect_used_paths(resolved_dataset_dir: Path) -> set[str]:
    used: set[str] = set()
    for split in ("train", "valid", "test"):
        p = resolved_dataset_dir / split / "_annotations.coco.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for im in data.get("images", []):
            fn = str(im.get("file_name", "")).strip()
            if fn:
                used.add(str(Path(fn).resolve()))
    return used


def _build_unlabeled_pool(images_root: Path, excluded_paths: set[str], cache_path: Path) -> list[str]:
    if cache_path.exists():
        try:
            js = json.loads(cache_path.read_text(encoding="utf-8"))
            items = js.get("unlabeled_paths", [])
            if isinstance(items, list) and items:
                return [str(x) for x in items]
        except Exception:
            pass

    paths: list[str] = []
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            rp = str(p.resolve())
            if rp not in excluded_paths:
                paths.append(rp)
    paths.sort()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"unlabeled_paths": paths}, indent=2), encoding="utf-8")
    return paths


def _clip_box_xyxy(box: np.ndarray, w: int, h: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    x1 = max(0.0, min(x1, float(w - 1)))
    y1 = max(0.0, min(y1, float(h - 1)))
    x2 = max(0.0, min(x2, float(w)))
    y2 = max(0.0, min(y2, float(h)))
    bw = x2 - x1
    bh = y2 - y1
    if bw < 1.0 or bh < 1.0:
        return None
    return x1, y1, bw, bh

def _build_pseudo_dataset(
    labeled_dataset_dir: Path,
    pseudo_out_dir: Path,
    teacher_ckpt: Path,
    model_cls,
    resolution: int,
    unlabeled_pool: list[str],
    seed: int,
) -> dict:
    train_json = labeled_dataset_dir / "train" / "_annotations.coco.json"
    if not train_json.exists():
        raise FileNotFoundError(f"Missing train annotations: {train_json}")
    src_train = json.loads(train_json.read_text(encoding="utf-8"))
    categories = src_train.get("categories", [])
    if len(categories) != 1:
        raise ValueError(f"Expected single class, got {len(categories)} classes")
    cat_id = int(categories[0]["id"])

    labeled_images = list(src_train.get("images", []))
    labeled_anns = list(src_train.get("annotations", []))

    target_pseudo_images = max(1, RFDETR_SS_PSEUDO_IMAGES_PER_LABELED * len(labeled_images))
    if RFDETR_SS_UNLABELED_MAX_IMAGES > 0:
        target_pseudo_images = min(target_pseudo_images, RFDETR_SS_UNLABELED_MAX_IMAGES)
    target_pseudo_images = min(target_pseudo_images, len(unlabeled_pool))

    candidates = list(unlabeled_pool)
    random.Random(seed).shuffle(candidates)
    chosen = candidates[:target_pseudo_images]

    pseudo_out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("valid", "test"):
        src = labeled_dataset_dir / split / "_annotations.coco.json"
        if src.exists():
            dst = pseudo_out_dir / split
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "_annotations.coco.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    kwargs: dict[str, Any] = {"pretrain_weights": str(teacher_ckpt)}
    if resolution > 0:
        kwargs["resolution"] = int(resolution)
    model = model_cls(**kwargs)
    if hasattr(model, "optimize_for_inference"):
        try:
            model.optimize_for_inference()
        except Exception:
            pass

    next_img_id = 1 + max([int(im.get("id", 0)) for im in labeled_images] + [0])
    next_ann_id = 1 + max([int(an.get("id", 0)) for an in labeled_anns] + [0])

    out_images = list(labeled_images)
    out_anns = list(labeled_anns)

    pseudo_image_count = 0
    pseudo_image_with_dets = 0
    pseudo_ann_count = 0

    for idx, img_path_s in enumerate(chosen, start=1):
        img_path = Path(img_path_s)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        w, h = image.size
        try:
            boxes, scores, _labels = _predict_one_image(model, image, RFDETR_SS_PSEUDO_SCORE_THRESH)
        except Exception:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)

        if len(scores) > 0 and RFDETR_SS_PSEUDO_MAX_DETS_PER_IMAGE > 0:
            keep = np.argsort(-scores)[: RFDETR_SS_PSEUDO_MAX_DETS_PER_IMAGE]
            boxes = boxes[keep]
            scores = scores[keep]

        image_id = next_img_id
        next_img_id += 1
        out_images.append(
            {
                "id": int(image_id),
                "file_name": str(img_path.resolve()),
                "width": int(w),
                "height": int(h),
                "is_pseudo": 1,
            }
        )
        pseudo_image_count += 1

        dets_this_image = 0
        for b, s in zip(boxes, scores):
            clipped = _clip_box_xyxy(b, w, h)
            if clipped is None:
                continue
            x, y, bw, bh = clipped
            out_anns.append(
                {
                    "id": int(next_ann_id),
                    "image_id": int(image_id),
                    "category_id": int(cat_id),
                    "bbox": [float(x), float(y), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                    "is_pseudo": 1,
                    "pseudo_score": float(s),
                }
            )
            next_ann_id += 1
            pseudo_ann_count += 1
            dets_this_image += 1

        if dets_this_image > 0:
            pseudo_image_with_dets += 1

        if idx % 100 == 0 or idx == len(chosen):
            print(
                f"[PSEUDO] progress {idx}/{len(chosen)} | anns={pseudo_ann_count} images_with_dets={pseudo_image_with_dets}"
            )

    train_out = {
        "images": out_images,
        "annotations": out_anns,
        "categories": categories,
    }
    train_dir = pseudo_out_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "_annotations.coco.json").write_text(json.dumps(train_out), encoding="utf-8")

    report = {
        "teacher_ckpt": str(teacher_ckpt),
        "candidate_unlabeled": len(unlabeled_pool),
        "selected_unlabeled": len(chosen),
        "pseudo_images_added": pseudo_image_count,
        "pseudo_images_with_dets": pseudo_image_with_dets,
        "pseudo_annotations_added": pseudo_ann_count,
        "total_train_images": len(out_images),
        "total_train_annotations": len(out_anns),
        "dataset_dir": str(pseudo_out_dir),
    }
    (pseudo_out_dir / "pseudo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _frac_tag(frac: float) -> str:
    return str(f"{float(frac):.5f}").rstrip("0").rstrip(".").replace(".", "p")


def _run_one_pair(
    session_root: Path,
    target_name: str,
    dataset_dir: Path,
    model_cls,
    init_mode: str,
    ssl_ckpt: str | None,
    epochs: int,
    lr: float,
    frac: float,
    fraction_seed: int,
    seed: int,
    unlabeled_pool: list[str],
    resolution: int,
) -> dict:
    pair_dir = session_root / f"frac_{_frac_tag(frac)}_fseed_{fraction_seed}_seed_{seed}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    sup_dir = pair_dir / "SUPERVISED"
    semi_dir = pair_dir / "SEMI_SUPERVISED"
    pseudo_data_dir = pair_dir / "_pseudo_dataset"

    t0 = time.time()
    print(f"[RUN] SUP start frac={frac} seed={seed} -> {sup_dir}")
    sup_best = base.train_one_run(
        target_name=target_name,
        dataset_dir=dataset_dir,
        out_dir=sup_dir,
        model_cls=model_cls,
        init_mode=init_mode,
        epochs=epochs,
        lr=lr,
        batch_size=RFDETR_SS_BATCH,
        grad_accum_steps=RFDETR_SS_GRAD_ACCUM,
        num_queries=RFDETR_SS_NUM_QUERIES,
        aug_copies=RFDETR_SS_AUG_COPIES,
        weight_decay=RFDETR_SS_WEIGHT_DECAY,
        dropout=RFDETR_SS_DROPOUT,
        early_stopping=RFDETR_SS_EARLY_STOPPING,
        early_stopping_patience=RFDETR_SS_EARLY_PATIENCE,
        early_stopping_min_delta=RFDETR_SS_EARLY_MIN_DELTA,
        early_stopping_use_ema=RFDETR_SS_EARLY_USE_EMA,
        backbone_ckpt=ssl_ckpt,
        train_fraction=float(frac),
        fraction_seed=int(fraction_seed),
        seed=int(seed),
        num_workers=base.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    sup_seconds = round(time.time() - t0, 2)

    teacher_ckpt = _find_teacher_checkpoint(sup_dir)

    resolved_full = base.build_resolved_static_dataset(
        src_dir=dataset_dir,
        dst_dir=base.OUTPUT_ROOT / f"{dataset_dir.name}_RESOLVED",
    )
    labeled_subset = base.build_fractional_train_split(
        resolved_root=resolved_full,
        frac=float(frac),
        cache_root=base.OUTPUT_ROOT,
        seed=int(fraction_seed),
    )

    t1 = time.time()
    pseudo_report = _build_pseudo_dataset(
        labeled_dataset_dir=labeled_subset,
        pseudo_out_dir=pseudo_data_dir,
        teacher_ckpt=teacher_ckpt,
        model_cls=model_cls,
        resolution=resolution,
        unlabeled_pool=unlabeled_pool,
        seed=seed,
    )
    pseudo_seconds = round(time.time() - t1, 2)

    if RFDETR_SS_REQUIRE_PSEUDO and int(pseudo_report["pseudo_annotations_added"]) <= 0:
        raise RuntimeError("Pseudo-labeling produced zero annotations. Lower RFDETR_SS_PSEUDO_SCORE_THRESH.")

    t2 = time.time()
    print(f"[RUN] SEMI start frac={frac} seed={seed} -> {semi_dir}")
    semi_best = base.train_one_run(
        target_name=target_name,
        dataset_dir=pseudo_data_dir,
        out_dir=semi_dir,
        model_cls=model_cls,
        init_mode=init_mode,
        epochs=epochs,
        lr=lr,
        batch_size=RFDETR_SS_BATCH,
        grad_accum_steps=RFDETR_SS_GRAD_ACCUM,
        num_queries=RFDETR_SS_NUM_QUERIES,
        aug_copies=0,
        weight_decay=RFDETR_SS_WEIGHT_DECAY,
        dropout=RFDETR_SS_DROPOUT,
        early_stopping=RFDETR_SS_EARLY_STOPPING,
        early_stopping_patience=RFDETR_SS_EARLY_PATIENCE,
        early_stopping_min_delta=RFDETR_SS_EARLY_MIN_DELTA,
        early_stopping_use_ema=RFDETR_SS_EARLY_USE_EMA,
        backbone_ckpt=ssl_ckpt,
        train_fraction=1.0,
        fraction_seed=int(fraction_seed),
        seed=int(seed),
        num_workers=base.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
    )
    semi_seconds = round(time.time() - t2, 2)

    row = {
        "target": target_name,
        "init_mode": init_mode,
        "model_cls": MODEL_CLS_NAME,
        "fraction": float(frac),
        "fraction_seed": int(fraction_seed),
        "seed": int(seed),
        "teacher_ckpt": str(teacher_ckpt),
        "supervised_map50": _round(sup_best.get("map50")),
        "supervised_map5095": _round(sup_best.get("map5095")),
        "semi_map50": _round(semi_best.get("map50")),
        "semi_map5095": _round(semi_best.get("map5095")),
        "delta_map50_semi_minus_supervised": (
            None if (sup_best.get("map50") is None or semi_best.get("map50") is None)
            else _round(float(semi_best["map50"]) - float(sup_best["map50"]))
        ),
        "delta_map5095_semi_minus_supervised": (
            None if (sup_best.get("map5095") is None or semi_best.get("map5095") is None)
            else _round(float(semi_best["map5095"]) - float(sup_best["map5095"]))
        ),
        "pseudo_images_added": int(pseudo_report["pseudo_images_added"]),
        "pseudo_images_with_dets": int(pseudo_report["pseudo_images_with_dets"]),
        "pseudo_annotations_added": int(pseudo_report["pseudo_annotations_added"]),
        "supervised_seconds": float(sup_seconds),
        "pseudo_seconds": float(pseudo_seconds),
        "semi_seconds": float(semi_seconds),
        "pair_dir": str(pair_dir),
        "supervised_dir": str(sup_dir),
        "semi_dir": str(semi_dir),
    }
    (pair_dir / "pair_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row

def main() -> None:
    model_cls = _resolve_model_cls(MODEL_CLS_NAME)

    if TARGET_KEY == "epi":
        target_name = "Squamous Epithelial Cell"
        dataset_dir = base.DATASET_EPI
        ssl_ckpt = os.getenv("RFDETR_EPI_SSL_CKPT", "").strip() or None
        epochs = RFDETR_SS_EPI_EPOCHS
        lr = RFDETR_SS_EPI_LR
    else:
        target_name = "Leucocyte"
        dataset_dir = base.DATASET_LEUCO
        ssl_ckpt = os.getenv("RFDETR_LEU_SSL_CKPT", "").strip() or None
        epochs = RFDETR_SS_LEU_EPOCHS
        lr = RFDETR_SS_LEU_LR

    if INIT_MODE == "ssl" and not ssl_ckpt:
        raise ValueError(f"INIT_MODE=ssl requires RFDETR_{TARGET_KEY.upper()}_SSL_CKPT")

    resolution = int(base.SEARCH_RESOLUTION)
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = base.OUTPUT_ROOT / f"session_semisup_{session_id}"
    session_root.mkdir(parents=True, exist_ok=False)

    print("[SEMI] session_root:", session_root)
    print("[SEMI] target:", target_name)
    print("[SEMI] dataset_dir:", dataset_dir)
    print("[SEMI] init_mode:", INIT_MODE)
    print("[SEMI] model_cls:", MODEL_CLS_NAME)
    print("[SEMI] fractions:", TRAIN_FRACTIONS)
    print("[SEMI] seeds:", SEEDS)
    print("[SEMI] fraction_seeds:", FRACTION_SEEDS)
    print("[SEMI] pseudo_score_thresh:", RFDETR_SS_PSEUDO_SCORE_THRESH)
    print("[SEMI] pseudo_max_dets_per_image:", RFDETR_SS_PSEUDO_MAX_DETS_PER_IMAGE)
    print("[SEMI] pseudo_images_per_labeled:", RFDETR_SS_PSEUDO_IMAGES_PER_LABELED)
    print("[SEMI] unlabeled_max_images:", RFDETR_SS_UNLABELED_MAX_IMAGES)

    resolved_full = base.build_resolved_static_dataset(
        src_dir=dataset_dir,
        dst_dir=base.OUTPUT_ROOT / f"{dataset_dir.name}_RESOLVED",
    )
    used_paths = _collect_used_paths(resolved_full)
    unlabeled_cache = session_root / "_cache" / f"unlabeled_pool_{dataset_dir.name}.json"
    unlabeled_pool = _build_unlabeled_pool(base.IMAGES_FALLBACK_ROOT, used_paths, unlabeled_cache)
    if not unlabeled_pool:
        raise RuntimeError(f"No unlabeled images found in {base.IMAGES_FALLBACK_ROOT} after excluding labeled paths")

    fs_by_seed = {s: FRACTION_SEEDS[0] for s in SEEDS} if len(FRACTION_SEEDS) == 1 else {s: FRACTION_SEEDS[i] for i, s in enumerate(SEEDS)}

    plan_rows = []
    for frac in TRAIN_FRACTIONS:
        for seed in SEEDS:
            plan_rows.append(
                {
                    "target": target_name,
                    "init_mode": INIT_MODE,
                    "fraction": float(frac),
                    "seed": int(seed),
                    "fraction_seed": int(fs_by_seed[seed]),
                    "model_cls": MODEL_CLS_NAME,
                    "epochs": int(epochs),
                    "lr": float(lr),
                    "batch": int(RFDETR_SS_BATCH),
                    "grad_accum_steps": int(RFDETR_SS_GRAD_ACCUM),
                    "num_queries": int(RFDETR_SS_NUM_QUERIES),
                }
            )

    (session_root / "TRAINING_PLAN.json").write_text(json.dumps(plan_rows, indent=2), encoding="utf-8")
    with (session_root / "TRAINING_PLAN.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(plan_rows[0].keys()))
        w.writeheader()
        for row in plan_rows:
            w.writerow(row)

    rows: list[dict] = []
    errors: list[dict] = []

    for idx, row in enumerate(plan_rows, start=1):
        frac = float(row["fraction"])
        seed = int(row["seed"])
        fraction_seed = int(row["fraction_seed"])
        print(f"[SEMI] === pair {idx}/{len(plan_rows)}: frac={frac} seed={seed} fraction_seed={fraction_seed} ===")
        try:
            result = _run_one_pair(
                session_root=session_root,
                target_name=target_name,
                dataset_dir=dataset_dir,
                model_cls=model_cls,
                init_mode=INIT_MODE,
                ssl_ckpt=ssl_ckpt,
                epochs=epochs,
                lr=lr,
                frac=frac,
                fraction_seed=fraction_seed,
                seed=seed,
                unlabeled_pool=unlabeled_pool,
                resolution=resolution,
            )
            rows.append(result)
        except Exception as e:
            err = {"fraction": frac, "seed": seed, "fraction_seed": fraction_seed, "error": repr(e)}
            errors.append(err)
            print(f"[SEMI][ERROR] {err}")
            if not RFDETR_SS_CONTINUE_ON_ERROR:
                raise

    summary = {
        "session_root": str(session_root),
        "target": target_name,
        "dataset_dir": str(dataset_dir),
        "init_mode": INIT_MODE,
        "model_cls": MODEL_CLS_NAME,
        "fractions": TRAIN_FRACTIONS,
        "seeds": SEEDS,
        "fraction_seeds": FRACTION_SEEDS,
        "pseudo_config": {
            "score_thresh": RFDETR_SS_PSEUDO_SCORE_THRESH,
            "max_dets_per_image": RFDETR_SS_PSEUDO_MAX_DETS_PER_IMAGE,
            "pseudo_images_per_labeled": RFDETR_SS_PSEUDO_IMAGES_PER_LABELED,
            "unlabeled_max_images": RFDETR_SS_UNLABELED_MAX_IMAGES,
            "require_pseudo": RFDETR_SS_REQUIRE_PSEUDO,
        },
        "n_completed_pairs": len(rows),
        "n_errors": len(errors),
        "errors": errors,
    }

    if rows:
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                -(r["semi_map50"] if r["semi_map50"] is not None else -1.0),
                -(r["supervised_map50"] if r["supervised_map50"] is not None else -1.0),
            ),
        )
        with (session_root / "semi_vs_supervised_leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
            w.writeheader()
            for r in rows_sorted:
                w.writerow(r)

        aggregate = []
        for frac in TRAIN_FRACTIONS:
            chunk = [r for r in rows if abs(float(r["fraction"]) - float(frac)) < 1e-12]
            sup_vals = [float(r["supervised_map50"]) for r in chunk if r.get("supervised_map50") is not None]
            semi_vals = [float(r["semi_map50"]) for r in chunk if r.get("semi_map50") is not None]
            delta_vals = [float(r["delta_map50_semi_minus_supervised"]) for r in chunk if r.get("delta_map50_semi_minus_supervised") is not None]
            aggregate.append(
                {
                    "fraction": float(frac),
                    "n_runs": int(len(chunk)),
                    "supervised_map50_mean": _round(float(np.mean(sup_vals))) if sup_vals else None,
                    "supervised_map50_std": _round(float(np.std(sup_vals))) if sup_vals else None,
                    "semi_map50_mean": _round(float(np.mean(semi_vals))) if semi_vals else None,
                    "semi_map50_std": _round(float(np.std(semi_vals))) if semi_vals else None,
                    "delta_map50_mean": _round(float(np.mean(delta_vals))) if delta_vals else None,
                    "delta_map50_std": _round(float(np.std(delta_vals))) if delta_vals else None,
                }
            )
        (session_root / "aggregate_by_fraction.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

        summary["best_by_semi_map50"] = rows_sorted[0]
        summary["aggregate_by_fraction"] = aggregate

    (session_root / "FINAL_SEMI_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[SEMI] Final summary:", session_root / "FINAL_SEMI_SUMMARY.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
