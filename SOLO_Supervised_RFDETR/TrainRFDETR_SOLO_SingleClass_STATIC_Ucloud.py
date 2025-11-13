# train_hpo_both_static_ovr.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from inspect import signature
import os, json, glob, os.path as op, itertools, time, csv, re
import multiprocessing as mp
from shutil import rmtree

import numpy as np, cv2
import albumentations as A

# ───────────────────────────────────────────────────────────────────────────────
# UCloud-friendly path detection
def _detect_user_base() -> str | None:
    aau = glob.glob("/work/Member Files:*")
    if aau: return op.basename(aau[0])
    sdu = [d for d in glob.glob("/work/*#*") if op.isdir(d)]
    return op.basename(sdu[0]) if sdu else None

USER_BASE_DIR = os.environ.get("USER_BASE_DIR") or _detect_user_base() or ""
if USER_BASE_DIR:
    os.environ["USER_BASE_DIR"] = USER_BASE_DIR
WORK_ROOT = Path("/work") / USER_BASE_DIR if USER_BASE_DIR else Path.cwd()

def env_path(name: str, default: Path) -> Path:
    v = os.getenv(name, "").strip()
    return Path(v) if v else default

# ───────────────────────────────────────────────
# Auto-detect GPU count and set MAX_PARALLEL (lazy torch import)
# ───────────────────────────────────────────────
def detect_visible_gpus() -> list[int]:
    """Return list of visible GPU indices from CUDA_VISIBLE_DEVICES or torch.cuda.device_count()."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        try:
            ids = [int(x) for x in visible.split(",") if x.strip() != ""]
            if ids:
                return ids
        except Exception:
            pass
    try:
        import torch  # lazy
        n = torch.cuda.device_count()
        return list(range(n))
    except Exception:
        return []

VISIBLE_GPUS = detect_visible_gpus()
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", str(len(VISIBLE_GPUS) if VISIBLE_GPUS else 1)))
print(f"[GPU DETECT] Visible GPUs: {VISIBLE_GPUS}  |  MAX_PARALLEL={MAX_PARALLEL}")

# Where the **real images** live (on your drive)
IMAGES_FALLBACK_ROOT = env_path(
    "IMAGES_FALLBACK_ROOT",
    WORK_ROOT / "CellScanData" / "Zoom10x - Quality Assessment_Cleaned",
)

# ───────────────────────────────────────────────────────────────────────────────
# ====== STATIC OVR DATASETS (jsons live in repo / project tree) ======
REPO_ROOT = Path.cwd()  # you usually run this from /work/projects/myproj
DEFAULT_ROOT = env_path(
    "STAT_DATASETS_ROOT",
    REPO_ROOT / "Stat_Dataset"
)

DATASET_LEUCO = Path(os.getenv(
    "DATASET_LEUCO",
    str(DEFAULT_ROOT / "QA-2025v1_Leucocyte_OVR")
))
DATASET_EPI = Path(os.getenv(
    "DATASET_EPI",
    str(DEFAULT_ROOT / "QA-2025v1_SquamousEpithelialCell_OVR")
))

def _autofind_dataset(root: Path, token: str) -> Path:
    cands = sorted([p for p in root.glob(f"*{token}*") if p.is_dir()])
    if not cands:
        raise FileNotFoundError(f"Could not find dataset for token '{token}' under {root}")
    return cands[-1]

if not DATASET_LEUCO.exists():
    DATASET_LEUCO = _autofind_dataset(DEFAULT_ROOT, "Leucocyte_OVR")
if not DATASET_EPI.exists():
    DATASET_EPI   = _autofind_dataset(DEFAULT_ROOT, "SquamousEpithelialCell_OVR")

# Where to put all outputs (AUG caches + HPO runs + leaderboards + final selections) → on the drive
OUTPUT_ROOT = env_path("OUTPUT_ROOT", WORK_ROOT / "RFDETR_SOLO_OUTPUT" / "HPO_BOTH_OVR")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))
SEED        = int(os.getenv("SEED", "42"))

# ───────────────────────────────────────────────────────────────────────────────
# Augmentation builder (TRAIN only). Val/Test pass through. Use Affine.
def _mk_pipeline_for(target_name: str):
    is_leuco = target_name.lower().startswith("leuco")
    if is_leuco:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                     scale=(1.10, 1.35), rotate=(-7, 7),
                     fit_output=False, cval=0, mode=cv2.BORDER_REFLECT_101, p=0.8),
            A.GaussNoise(p=0.25),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.01, p=0.4),
            A.GaussianBlur(blur_limit=(3, 3), p=0.10),
        ], bbox_params=A.BboxParams(format="coco", label_fields=["category_id"], min_visibility=0.30))
    else:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                     scale=(0.90, 1.10), rotate=(-5, 5),
                     fit_output=False, cval=0, mode=cv2.BORDER_REFLECT_101, p=0.7),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.GaussNoise(p=0.2),
        ], bbox_params=A.BboxParams(format="coco", label_fields=["category_id"], min_visibility=0.40))

def _read_json(p: Path): return json.loads(p.read_text(encoding="utf-8"))
def _write_json(p: Path, obj): p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
def _safe_imread(path: Path):
    from PIL import Image
    return np.array(Image.open(str(path)).convert("RGB"))

def _clip_bbox(b, w, h):
    x, y, bw, bh = b
    x = max(0.0, min(float(x),  float(w - 1)))
    y = max(0.0, min(float(y),  float(h - 1)))
    bw = max(1.0, min(float(bw), float(w - x)))
    bh = max(1.0, min(float(bh), float(h - y)))
    return [x, y, bw, bh]

_WINDOWS_PATH_RE = re.compile(r"^[a-zA-Z]:\\")
def _resolve_img_fp(src_dir: Path, file_name: str) -> Path:
    """
    Resolve an image path from COCO 'file_name' robustly:
      1) literal path
      2) relative to src_dir/train
      3) scan IMAGES_FALLBACK_ROOT by basename (your drive)
    """
    p = Path(file_name)
    if p.exists():
        return p

    alt = (src_dir / "train" / file_name).resolve()
    if alt.exists():
        return alt

    base = Path(file_name).name

    if IMAGES_FALLBACK_ROOT.exists():
        hits = list(IMAGES_FALLBACK_ROOT.rglob(base))
        if len(hits) == 1:
            return hits[0]
        elif len(hits) > 1:
            hits.sort(key=lambda q: len(str(q)))
            return hits[0]

    if _WINDOWS_PATH_RE.match(file_name) and IMAGES_FALLBACK_ROOT.exists():
        hits = list(IMAGES_FALLBACK_ROOT.rglob(base))
        if hits:
            hits.sort(key=lambda q: len(str(q)))
            return hits[0]

    raise FileNotFoundError(
        f"Could not resolve image path for '{file_name}'. "
        f"Tried literal, relative to {src_dir/'train'}, and fallback {IMAGES_FALLBACK_ROOT}."
    )

# ───────────────────────────────────────────────────────────────────────────────
# Build AUG dataset with OK-marker; rebuild if incomplete
import time
from tqdm import tqdm  # progress bars

def build_augmented_train_set(src_dir: Path, dst_dir: Path, target_name: str,
                              copies_per_image: int,
                              keep_originals: bool = True,
                              jpeg_quality: int = 92) -> Path:
    """
    Build a per-class augmentation cache under `dst_dir`.

    - Train: read COCO json from `src_dir/train/_annotations.coco.json`,
      copy originals + create augmented images into `dst_dir/train/images`,
      write a new COCO json for train.
    - Valid/Test: copy COCO jsons AND the corresponding images from the
      original dataset into `dst_dir/valid` and `dst_dir/test` with the
      same relative paths as in the json.

    The cache is considered complete when:
      * .AUG_OK exists
      * train/images has files
      * valid/ and test/ each contain at least one image file.
    """
    assert (src_dir / "train" / "_annotations.coco.json").exists(), f"Missing train json in {src_dir}"

    ok_marker = dst_dir / ".AUG_OK"
    train_img_dir = dst_dir / "train" / "images"

    # ---- helper: does this directory contain any files? ----
    def _has_any_images(p: Path) -> bool:
        return p.exists() and any(p.rglob("*.*"))

    # ---- reuse existing cache if complete ----
    if dst_dir.exists():
        valid_ok = _has_any_images(dst_dir / "valid")
        test_ok  = _has_any_images(dst_dir / "test")
        train_ok = _has_any_images(train_img_dir)
        if ok_marker.exists() and train_ok and valid_ok and test_ok:
            print(f"[AUG] Using existing AUG dir: {dst_dir}")
            return dst_dir
        else:
            print(f"[AUG] Incomplete AUG dir detected, rebuilding: {dst_dir}")
            rmtree(dst_dir, ignore_errors=True)

    print(f"[AUG] Build train AUG for '{target_name}' → {dst_dir}")
    t0 = time.time()

    # ─────────────────────────────────────────────
    # 1) Passthrough VALID + TEST (copy json + images)
    # ─────────────────────────────────────────────
    for part in ("valid", "test"):
        src_json = src_dir / part / "_annotations.coco.json"
        if not src_json.exists():
            print(f"[AUG][WARN] No {part} json in {src_dir}, skipping {part} passthrough.")
            continue

        dst_part = dst_dir / part
        dst_part.mkdir(parents=True, exist_ok=True)

        ann_part = _read_json(src_json)
        _write_json(dst_part / "_annotations.coco.json", ann_part)

        images = ann_part.get("images", [])
        print(f"[AUG] Passthrough {part}: {len(images)} images")

        for im in images:
            rel = Path(im["file_name"])  # e.g. "Sample 14/Patches for Sample 14/xxx.tif"
            src_fp = (src_dir / part / rel).resolve()
            if not src_fp.exists():
                print(f"[AUG][WARN] Missing source image for {part}: {src_fp}")
                continue

            dst_fp = dst_part / rel
            dst_fp.parent.mkdir(parents=True, exist_ok=True)
            if not dst_fp.exists():
                copy2(src_fp, dst_fp)

    # ─────────────────────────────────────────────
    # 2) TRAIN: originals + augmentations
    # ─────────────────────────────────────────────
    ann_train = _read_json(src_dir / "train" / "_annotations.coco.json")
    cats      = ann_train["categories"]
    images    = {im["id"]: im for im in ann_train["images"]}

    anns_by_img = defaultdict(list)
    for a in ann_train["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    out_images: list[dict] = []
    out_anns:   list[dict] = []
    next_img_id = 1
    next_ann_id = 1

    pipeline    = _mk_pipeline_for(target_name)
    train_root  = dst_dir / "train"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    out_ann_path = train_root / "_annotations.coco.json"

    total_images = len(images)
    total_aug    = total_images * copies_per_image
    print(f"[AUG] {total_images} base train images × {copies_per_image} augmentations each = {total_aug} new images")

    # ---- 2a) keep originals in train/images ----
    if keep_originals:
        for iid, im in tqdm(images.items(), desc="[AUG] Writing train originals", ncols=100):
            new_iid = next_img_id
            next_img_id += 1

            # robust resolution (handles Windows paths etc.)
            src_fp = _resolve_img_fp(src_dir, im["file_name"])
            ext = Path(src_fp).suffix.lower()
            if ext not in (".png", ".tif", ".tiff"):
                ext = ".jpg"

            new_name = f"img_{new_iid}{ext}"
            dst_fp   = train_img_dir / new_name

            img_np = _safe_imread(src_fp)
            if ext == ".png":
                cv2.imwrite(
                    str(dst_fp),
                    cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                )
            else:
                cv2.imwrite(
                    str(dst_fp),
                    cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )

            # file_name relative to train root (so root/train + file_name works)
            rel_name = dst_fp.relative_to(train_root)
            out_images.append({
                "id": new_iid,
                "file_name": str(rel_name),
                "width":  int(im["width"]),
                "height": int(im["height"]),
            })

            for a in anns_by_img.get(iid, []):
                aa = dict(a)
                aa["id"]       = next_ann_id
                aa["image_id"] = new_iid
                next_ann_id   += 1
                aa["bbox"]     = _clip_bbox(aa["bbox"], im["width"], im["height"])
                out_anns.append(aa)

    # ---- 2b) augmented copies with progress bar ----
    pbar = tqdm(total=total_images, desc=f"[AUG] Generating {copies_per_image}× per train image", ncols=100)
    for iid, im in images.items():
        src_fp = _resolve_img_fp(src_dir, im["file_name"])
        img_np = _safe_imread(src_fp)

        orig_bboxes = []
        orig_labels = []
        for a in anns_by_img.get(iid, []):
            x, y, w, h = a["bbox"]
            orig_bboxes.append([float(x), float(y), float(w), float(h)])
            orig_labels.append(a["category_id"])

        for k in range(copies_per_image):
            t = pipeline(image=img_np, bboxes=orig_bboxes, category_id=orig_labels)
            aug = t["image"]
            bbs = t["bboxes"]
            lbs = t["category_id"]
            if not bbs:
                continue  # skip if augmentation dropped all boxes

            new_iid = next_img_id
            next_img_id += 1
            new_name = f"img_{new_iid}_aug{k+1}.jpg"
            dst_fp   = train_img_dir / new_name

            cv2.imwrite(
                str(dst_fp),
                cv2.cvtColor(aug, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

            H, W = aug.shape[:2]
            rel_name = dst_fp.relative_to(train_root)
            out_images.append({
                "id": new_iid,
                "file_name": str(rel_name),
                "width":  int(W),
                "height": int(H),
            })

            for bb, lab in zip(bbs, lbs):
                bb = _clip_bbox(bb, W, H)
                aa = {
                    "id":         next_ann_id,
                    "image_id":   new_iid,
                    "category_id": int(lab),
                    "bbox":       bb,
                    "area":       float(max(1.0, bb[2] * bb[3])),
                    "iscrowd":    0,
                }
                next_ann_id += 1
                out_anns.append(aa)

        pbar.update(1)
    pbar.close()

    # ---- 2c) write train json + OK marker ----
    _write_json(out_ann_path, {
        "images":      out_images,
        "annotations": out_anns,
        "categories":  cats,
    })
    ok_marker.write_text("ok", encoding="utf-8")

    elapsed = time.time() - t0
    print(f"[AUG] Train images: base={len(images)}, total_out={len(out_images)} | Done in {elapsed/60:.1f} min")
    return dst_dir

# Reusable per-class AUG cache (faster & fair for HPO)
def get_or_build_aug_cache(target_name: str, dataset_dir: Path, root_out: Path, aug_copies: int) -> Path:
    class_token = target_name.replace(" ", "")
    cache_dir = root_out / f"{class_token}_AUG_CACHE"
    return build_augmented_train_set(dataset_dir, cache_dir, target_name,
                                     copies_per_image=aug_copies, keep_originals=True)

# ───────────────────────────────────────────────────────────────────────────────
# Training (single run). Called in worker processes.
def train_one_run(target_name: str,
                  dataset_dir: Path,
                  out_dir: Path,
                  model_cls,
                  resolution: int,
                  epochs: int,
                  lr: float,
                  lr_encoder_mult: float,
                  batch_size: int,
                  num_queries: int,
                  warmup_steps: int,
                  scale_range: tuple[float, float],
                  rot_deg: float,
                  color_jitter: float,
                  gauss_blur_prob: float,
                  aug_copies: int,
                  seed: int = 42,
                  num_workers: int = 8,
                  pin_memory: bool = True,
                  persistent_workers: bool = True) -> dict:

    # Use the per-class AUG cache instead of per-run AUG dirs
    aug_dir = get_or_build_aug_cache(target_name, dataset_dir, OUTPUT_ROOT, aug_copies)

    model = model_cls()
    sig = signature(model.train)
    can = set(sig.parameters.keys())

    kwargs = dict(
        dataset_dir=str(aug_dir),
        output_dir=str(out_dir),
        class_names=[target_name],

        resolution=resolution,
        batch_size=batch_size,
        grad_accum_steps=1,
        epochs=epochs,
        lr=lr,
        weight_decay=5e-4,
        dropout=0.1,
        num_queries=num_queries,

        multi_scale=True,
        gradient_checkpointing=True,
        amp=True,
        num_workers=min(num_workers, os.cpu_count() or num_workers),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,

        seed=seed,
        early_stopping=True,
        early_stopping_patience=35 if target_name.lower().startswith("leuco") else 20,
        early_stopping_metric="map_50_95",
        early_stopping_min_delta=0.001,
        early_stopping_use_ema=True,
        checkpoint_interval=1,
        run_test=True,

        # trainer-side augs (if supported)
        hflip_prob=0.5,
        rotation_degrees=rot_deg,
        scale_range=scale_range,
        color_jitter=color_jitter,
        gaussian_blur_prob=gauss_blur_prob,
        square_resize_div_64=True,
        random_resize_via_padding=True,
    )
    if "lr_encoder" in can: kwargs["lr_encoder"] = lr * lr_encoder_mult
    if "lr_schedule" in can: kwargs["lr_schedule"] = "cosine"
    if "warmup_steps" in can: kwargs["warmup_steps"] = warmup_steps
    if "clip_grad_norm" in can: kwargs["clip_grad_norm"] = 0.1
    if "ema" in can: kwargs["ema"] = True

    # Log meta
    meta = out_dir / "run_meta"; meta.mkdir(parents=True, exist_ok=True)
    (meta / "train_kwargs.json").write_text(json.dumps(kwargs, indent=2), encoding="utf-8")
    (meta / "dataset_info.json").write_text(json.dumps({
        "dataset_dir": str(dataset_dir), "dataset_dir_aug": str(aug_dir),
        "target_name": target_name
    }, indent=2), encoding="utf-8")

    print(f"[TRAIN] {model.__class__.__name__} — {target_name} → {out_dir}")
    model.train(**kwargs)

    best = find_best_val(out_dir)
    (out_dir / "val_best_summary.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best

def find_best_val(output_dir: Path) -> dict:
    candidates = [
        "val_best_summary.json",
        "val_metrics.json", "metrics_val.json", "coco_eval_val.json",
        "metrics.json", "val_results.json", "results_val.json"
    ]
    for name in candidates:
        p = output_dir / name
        if not p.exists(): continue
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
            def pick(keys):
                for k in keys:
                    if k in js: return float(js[k])
            return {
                "best_epoch": js.get("best") or js.get("best_epoch") or js.get("epoch"),
                "map50":      pick(["map50","mAP50","ap50","AP50","bbox/AP50"]),
                "map5095":    pick(["map","mAP","mAP5095","bbox/mAP"]),
                "source":     name
            }
        except Exception:
            continue
    return {"best_epoch": None, "map50": None, "map5095": None, "source": "not_found"}

# ───────────────────────────────────────────────────────────────────────────────
# HPO spaces (compact) — keep using MODEL_CLS but we’ll resolve it inside the worker
SEARCH_LEUCO = {
    "MODEL_CLS":       ["RFDETRLarge"],
    "RESOLUTION":      [672, 896],
    "EPOCHS":          [180, 260],
    "LR":              [8e-5, 1e-4],
    "LR_ENCODER_MULT": [0.10, 0.15],
    "BATCH":           [4],
    "NUM_QUERIES":     [300, 450],
    "WARMUP_STEPS":    [2000, 4000],
    "AUG_COPIES":      [2],
    "SCALE_RANGE":     [(1.0, 1.6)],
    "ROT_DEG":         [7.0],
    "COLOR_JITTER":    [0.25],
    "GAUSS_BLUR":      [0.10],
}

SEARCH_EPI = {
    "MODEL_CLS":       ["RFDETRLarge"],
    "RESOLUTION":      [640, 672],
    "EPOCHS":          [160, 220],
    "LR":              [1e-4],
    "LR_ENCODER_MULT": [0.15],
    "BATCH":           [6],
    "NUM_QUERIES":     [300],
    "WARMUP_STEPS":    [1500, 3000],
    "AUG_COPIES":      [1],
    "SCALE_RANGE":     [(0.9, 1.1)],
    "ROT_DEG":         [5.0],
    "COLOR_JITTER":    [0.20],
    "GAUSS_BLUR":      [0.20],
}

def grid(space: dict):
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

# ───────────────────────────────────────────────────────────────────────────────
# Trial-level parallel launcher with allocator + OOM backoff
def _get_visible_gpu_ids() -> list[int]:
    vis = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if vis:
        try:
            return [int(x) for x in vis.split(",") if x != ""]
        except ValueError:
            pass
    try:
        import torch  # lazy
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []

def _configure_allocator_env():
    """Configure safe allocator options *only if not set by user*."""
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if conf.strip() == "":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def _set_perf_toggles():
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def _oom_backoff(cfg: dict) -> dict:
    """Return a slightly 'lighter' config when OOM is detected."""
    new = dict(cfg)
    if new["BATCH"] > 2:
        new["BATCH"] = max(2, new["BATCH"] - 1)
        return new
    if new["NUM_QUERIES"] > 240:
        new["NUM_QUERIES"] = int(max(240, round(new["NUM_QUERIES"] * 0.8)))
        return new
    # drop resolution by ~10% (round to nearest 32)
    def _downscale(r):
        x = int(round(r * 0.9 / 32) * 32)
        return max(512, x)
    new["RESOLUTION"] = _downscale(new["RESOLUTION"])
    return new

def _worker_entry(cfg: dict, gpu_id: int, run_idx: int, target_name: str,
                  dataset_dir: str, out_root: str, result_q: mp.Queue):
    try:
        # Mask to a single physical GPU for this process BEFORE importing torch/rfdetr
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        _configure_allocator_env()

        import torch  # import AFTER masking
        torch.cuda.set_device(0)  # within this worker, the masked GPU is cuda:0

        # Import model classes AFTER masking
        from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge
        name2cls = {
            "RFDETRSmall": RFDETRSmall,
            "RFDETRMedium": RFDETRMedium,
            "RFDETRLarge": RFDETRLarge,
        }
        # Accept either a class or a string in SEARCH_*["MODEL_CLS"]
        mc = cfg["MODEL_CLS"]
        if isinstance(mc, str):
            model_cls = name2cls[mc]
        else:
            # if user passed the class itself, remap to the class again (no-op) but avoids pickling issues
            model_cls = name2cls.get(getattr(mc, "__name__", "RFDETRLarge"), RFDETRLarge)

        # Helpful mapping print
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        cur = torch.cuda.current_device()
        print(f"[WORKER {run_idx:03d}] Physical GPU {gpu_id} masked as cuda:0 | "
              f"CUDA_VISIBLE_DEVICES={vis} | using cuda:{cur} ({torch.cuda.get_device_name(cur)})")

        _set_perf_toggles()

        out_root = Path(out_root)
        run_dir = out_root / f"run_{run_idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # First attempt
        t0 = time.time()
        try_cfg = dict(cfg)
        try:
            best = train_one_run(
                target_name=target_name,
                dataset_dir=Path(dataset_dir),
                out_dir=run_dir,
                model_cls=model_cls,
                resolution=try_cfg["RESOLUTION"],
                epochs=try_cfg["EPOCHS"],
                lr=try_cfg["LR"],
                lr_encoder_mult=try_cfg["LR_ENCODER_MULT"],
                batch_size=try_cfg["BATCH"],
                num_queries=try_cfg["NUM_QUERIES"],
                warmup_steps=try_cfg["WARMUP_STEPS"],
                scale_range=try_cfg["SCALE_RANGE"],
                rot_deg=try_cfg["ROT_DEG"],
                color_jitter=try_cfg["COLOR_JITTER"],
                gauss_blur_prob=try_cfg["GAUSS_BLUR"],
                aug_copies=try_cfg["AUG_COPIES"],
                seed=SEED,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                persistent_workers=True,
            )
        except RuntimeError as e:
            msg = str(e)
            if "CUDA out of memory" in msg or "CUDNN_STATUS_ALLOC_FAILED" in msg:
                torch.cuda.empty_cache()
                try_cfg = _oom_backoff(try_cfg)
                (run_dir / "oom_backoff.json").write_text(json.dumps(try_cfg, indent=2), encoding="utf-8")
                best = train_one_run(
                    target_name=target_name,
                    dataset_dir=Path(dataset_dir),
                    out_dir=run_dir,
                    model_cls=model_cls,
                    resolution=try_cfg["RESOLUTION"],
                    epochs=try_cfg["EPOCHS"],
                    lr=try_cfg["LR"],
                    lr_encoder_mult=try_cfg["LR_ENCODER_MULT"],
                    batch_size=try_cfg["BATCH"],
                    num_queries=try_cfg["NUM_QUERIES"],
                    warmup_steps=try_cfg["WARMUP_STEPS"],
                    scale_range=try_cfg["SCALE_RANGE"],
                    rot_deg=try_cfg["ROT_DEG"],
                    color_jitter=try_cfg["COLOR_JITTER"],
                    gauss_blur_prob=try_cfg["GAUSS_BLUR"],
                    aug_copies=try_cfg["AUG_COPIES"],
                    seed=SEED,
                    num_workers=max(2, NUM_WORKERS // 2),
                    pin_memory=False,
                    persistent_workers=False,
                )
            else:
                raise

        dur = round(time.time() - t0, 2)

        row = {
            "run_idx": run_idx,
            "target": target_name,
            **{k: (v if not hasattr(v, "__name__") else v.__name__) for k, v in try_cfg.items()},
            "val_AP50": best.get("map50"),
            "val_mAP5095": best.get("map5095"),
            "best_epoch": best.get("best_epoch"),
            "metrics_source": best.get("source"),
            "output_dir": str(run_dir),
            "seconds": dur,
        }
        (run_dir / "hpo_record.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        result_q.put(("ok", row))
    except Exception as e:
        result_q.put(("err", {"run_idx": run_idx, "target": target_name, "error": repr(e)}))

def run_hpo_for_class(target_name: str, dataset_dir: Path, search_space: dict, root_out: Path) -> dict:
    """
    Parallel trial launcher across visible GPUs. Returns leaderboard + best.
    """
    class_token = target_name.replace(" ", "")
    out_root = root_out / class_token
    out_root.mkdir(parents=True, exist_ok=True)

    gpu_ids = _get_visible_gpu_ids()
    if not gpu_ids:
        raise RuntimeError("No visible GPUs. Set CUDA_VISIBLE_DEVICES or ensure CUDA is available.")
    max_parallel = max(1, min(MAX_PARALLEL, len(gpu_ids)))
    print(f"[HPO:{class_token}] Visible GPUs: {gpu_ids}  |  MAX_PARALLEL={max_parallel}")

    def grid(space: dict):
        keys = list(space.keys())
        vals = [space[k] for k in keys]
        for combo in itertools.product(*vals):
            yield dict(zip(keys, combo))

    jobs = list(grid(search_space))
    leaderboard = []

    ctx = mp.get_context("spawn")
    result_q: mp.Queue = ctx.Queue()
    active: dict[int, mp.Process] = {}
    next_gpu_slot = 0

    def launch_job(run_idx: int, cfg: dict):
        nonlocal next_gpu_slot
        gpu_id = gpu_ids[next_gpu_slot]
        next_gpu_slot = (next_gpu_slot + 1) % len(gpu_ids)
        p = ctx.Process(target=_worker_entry,
                        args=(cfg, gpu_id, run_idx, target_name, str(dataset_dir), str(out_root), result_q))
        p.daemon = False
        p.start()
        active[run_idx] = p
        print(f"[HPO:{class_token}] Launched run {run_idx:03d} on GPU {gpu_id}: {cfg}")

    run_idx = 0
    for cfg in jobs:
        while len([p for p in active.values() if p.is_alive()]) >= max_parallel:
            while not result_q.empty():
                status, payload = result_q.get()
                rid = payload.get("run_idx")
                if rid in active and not active[rid].is_alive():
                    active[rid].join()
                    del active[rid]
                if status == "ok":
                    leaderboard.append(payload)
                    print(f"[HPO:{class_token}] Finished run {payload['run_idx']:03d} — "
                          f"AP50={payload['val_AP50']} mAP={payload['val_mAP5095']}")
                else:
                    print(f"[HPO:{class_token}] ERROR run {payload.get('run_idx')}: {payload.get('error')}")
            time.sleep(1)

        run_idx += 1
        launch_job(run_idx, cfg)

    while active:
        while not result_q.empty():
            status, payload = result_q.get()
            rid = payload.get("run_idx")
            if rid in active and not active[rid].is_alive():
                active[rid].join()
                del active[rid]
            if status == "ok":
                leaderboard.append(payload)
                print(f"[HPO:{class_token}] Finished run {payload['run_idx']:03d} — "
                      f"AP50={payload['val_AP50']} mAP={payload['val_mAP5095']}")
            else:
                print(f"[HPO:{class_token}] ERROR run {payload.get('run_idx')}: {payload.get('error')}")
        time.sleep(1)

    if not leaderboard:
        return {"leaderboard": [], "best": {}, "out_dir": str(out_root)}

    def sort_key(r):
        a = r["val_AP50"]; b = r["val_mAP5095"]
        return (-(a if a is not None else -1), -(b if b is not None else -1))

    leaderboard.sort(key=sort_key)

    csv_path = out_root / "hpo_leaderboard.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(leaderboard[0].keys()))
        w.writeheader(); [w.writerow(r) for r in leaderboard]
    (out_root / "hpo_top.json").write_text(json.dumps(leaderboard[:5], indent=2), encoding="utf-8")

    best_row = leaderboard[0]
    (out_root / "hpo_best.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")
    return {"leaderboard": leaderboard, "best": best_row, "out_dir": str(out_root)}

# ───────────────────────────────────────────────────────────────────────────────
def main():
    print("[WORK_ROOT]", WORK_ROOT)
    print("[DATASETS]")
    print("  Leucocyte:", DATASET_LEUCO)
    print("  Epithelial:", DATASET_EPI)
    for p in (DATASET_LEUCO, DATASET_EPI):
        for part in ("train","valid"):
            if not (p / part / "_annotations.coco.json").exists():
                raise FileNotFoundError(f"Missing {part} split in {p}")

    # HPO for each class
    res_leu = run_hpo_for_class("Leucocyte", DATASET_LEUCO, SEARCH_LEUCO, OUTPUT_ROOT)
    res_epi = run_hpo_for_class("Squamous Epithelial Cell", DATASET_EPI, SEARCH_EPI, OUTPUT_ROOT)

    final = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "work_root": str(WORK_ROOT),
        "leucocyte": {
            "dataset": str(DATASET_LEUCO),
            "best": res_leu["best"],
            "leaderboard_top5": res_leu["leaderboard"][:5]
        },
        "epithelial": {
            "dataset": str(DATASET_EPI),
            "best": res_epi["best"],
            "leaderboard_top5": res_epi["leaderboard"][:5]
        }
    }
    (OUTPUT_ROOT / "FINAL_HPO_SUMMARY.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("\n[FINAL] Summary →", OUTPUT_ROOT / "FINAL_HPO_SUMMARY.json")
    print(json.dumps(final, indent=2))

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
