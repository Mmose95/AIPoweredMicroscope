from __future__ import annotations
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from inspect import signature
import os, json, glob, os.path as op, itertools, time, csv, multiprocessing as mp
import re
import random
from collections import defaultdict as ddict
from PIL import Image
import numpy as np

# ───────────────────────────────────────────────────────────────────────────────
# TOGGLES
# ───────────────────────────────────────────────────────────────────────────────
# Which classes to train in this HPO run:
#  - "leu"  -> only Leucocyte
#  - "epi"  -> only Squamous Epithelial Cell
#  - "all"  -> both
HPO_TARGET = os.environ.get("RFDETR_HPO_TARGET", "epi").lower()
BACKBONE_BLOCK_SIZE = 56


def _round_up_to_multiple(v: int, m: int) -> int:
    return ((v + m - 1) // m) * m

def _resolve_input_mode() -> tuple[bool, str, int, int]:
    mode_raw = os.getenv("RFDETR_INPUT_MODE", "").strip().lower()
    if mode_raw:
        aliases = {
            "patch": "224",
            "patch224": "224",
            "patch_224": "224",
            "full": "640",
            "full640": "640",
            "full_640": "640",
        }
        mode_norm = aliases.get(mode_raw, mode_raw)
        m = re.fullmatch(r"(\d+)(?:x\1)?", mode_norm)
        if not m:
            raise ValueError(
                f"Invalid RFDETR_INPUT_MODE={mode_raw!r}. "
                f"Use 640 for full-image mode, or any positive integer (e.g. 224) for patch mode."
            )
        mode_size = int(m.group(1))
        if mode_size <= 0:
            raise ValueError(f"RFDETR_INPUT_MODE must be a positive integer, got {mode_size}")

        use_patch = mode_size != 640
        patch_size = mode_size if use_patch else 224
        full_resolution = int(os.getenv("RFDETR_FULL_RESOLUTION", str(mode_size)))
        return use_patch, str(mode_size), patch_size, full_resolution

    # Backward-compatible path: keep old env behavior if RFDETR_INPUT_MODE is unset.
    use_patch = bool(int(os.getenv("RFDETR_USE_PATCH_224", "1")))
    patch_size = int(os.getenv("RFDETR_PATCH_SIZE", "224"))
    full_resolution = int(os.getenv("RFDETR_FULL_RESOLUTION", "640"))
    return use_patch, (str(patch_size) if use_patch else "640"), patch_size, full_resolution

# If RFDETR_INPUT_MODE != 640 we run patch mode with patch size = int(RFDETR_INPUT_MODE).
# If RFDETR_INPUT_MODE == 640 we run full-image mode.
USE_PATCH_224, INPUT_MODE, PATCH_SIZE, FULL_RESOLUTION = _resolve_input_mode()
if not USE_PATCH_224 and FULL_RESOLUTION % BACKBONE_BLOCK_SIZE != 0:
    old = FULL_RESOLUTION
    FULL_RESOLUTION = _round_up_to_multiple(FULL_RESOLUTION, BACKBONE_BLOCK_SIZE)
    print(
        f"[INPUT MODE][WARN] Full-mode resolution {old} is not divisible by "
        f"{BACKBONE_BLOCK_SIZE}; using {FULL_RESOLUTION}."
    )

print(f"[HPO] Target classes: {HPO_TARGET!r} (env RFDETR_HPO_TARGET)")
print(f"[INPUT MODE] RFDETR_INPUT_MODE={INPUT_MODE}  USE_PATCH_224={USE_PATCH_224}")
print(f"[INPUT SIZE] PATCH_SIZE={PATCH_SIZE}  FULL_RESOLUTION={FULL_RESOLUTION}")
if os.getenv("RFDETR_INPUT_MODE", "").strip():
    print("[INPUT MODE] RFDETR_INPUT_MODE is authoritative; legacy RFDETR_USE_PATCH_224/RFDETR_PATCH_SIZE are ignored.")
if PATCH_SIZE <= 0:
    raise ValueError(f"RFDETR_PATCH_SIZE must be > 0, got {PATCH_SIZE}")
if FULL_RESOLUTION <= 0:
    raise ValueError(f"RFDETR_FULL_RESOLUTION must be > 0, got {FULL_RESOLUTION}")

# ───────────────────────────────────────────────────────────────────────────────
# UCloud-friendly path detection
# ───────────────────────────────────────────────────────────────────────────────
def _detect_user_base() -> str | None:
    aau = glob.glob("/work/Member Files:*")
    if aau:
        return op.basename(aau[0])
    sdu = [d for d in glob.glob("/work/*#*") if op.isdir(d)]
    return op.basename(sdu[0]) if sdu else None

USER_BASE_DIR = os.environ.get("USER_BASE_DIR") or _detect_user_base() or ""
if USER_BASE_DIR:
    os.environ["USER_BASE_DIR"] = USER_BASE_DIR
WORK_ROOT = Path("/work") / USER_BASE_DIR if USER_BASE_DIR else Path.cwd()

def env_path(name: str, default: Path) -> Path:
    v = os.getenv(name, "").strip()
    return Path(v) if v else default

# ───────────────────────────────────────────────────────────────────────────────
# SSL backbones to probe (encoder checkpoints)
# ───────────────────────────────────────────────────────────────────────────────
SSL_CKPT_ROOT = env_path(
    "SSL_CKPT_ROOT",
    WORK_ROOT / "SSL_Checkpoints",
)

SSL_BACKBONES = [
    SSL_CKPT_ROOT / "epoch_epoch-004.ckpt",
    SSL_CKPT_ROOT / "epoch_epoch-009.ckpt",
    SSL_CKPT_ROOT / "epoch_epoch-014.ckpt",
    SSL_CKPT_ROOT / "epoch_epoch-029.ckpt", #This one is the "best" from long SSL training.
    SSL_CKPT_ROOT / "last.ckpt",
]

print("[SSL BACKBONES]")
for p in SSL_BACKBONES:
    print("  ", p)

BEST_SSL_CKPT = str(SSL_CKPT_ROOT / "epoch_epoch-029.ckpt")  # <- adjust to winner if needed

# ───────────────────────────────────────────────
# Path resolution helpers for COCO
# ───────────────────────────────────────────────
VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
WINDOWS_PATH_RE = re.compile(r"^[a-zA-Z]:\\")  # detect old Windows-style paths

def _index_image_paths(root: Path):
    """Index all images under root for fast lookup."""
    by_rel = {}
    by_name = ddict(list)
    root = root.resolve()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            rel = p.resolve().relative_to(root).as_posix()
            by_rel[rel] = p
            by_name[p.name].append(p)
    return by_rel, by_name


def _resolve_image_path(file_name: str, images_root: Path,
                        by_rel: dict, by_name: dict) -> Path:
    """
    Resolve a COCO file_name to an actual image file on the mounted drive.
    - Handles 'Sample 25/...tif'
    - Handles old Windows absolute paths
    """
    # normalize slashes
    rel = file_name.replace("\\", "/")

    # direct relative match under images_root
    direct = (images_root / rel)
    if direct.exists():
        return direct

    # look it up by relative key
    if rel in by_rel:
        return by_rel[rel]

    # fallback: by basename
    base = Path(rel).name
    if base in by_name:
        cands = by_name[base]
        if len(cands) == 1:
            return cands[0]
        # if multiple, pick shortest path
        cands.sort(key=lambda q: len(str(q)))
        return cands[0]

    # if JSON kept Windows absolute paths, strip drive + search by basename
    if WINDOWS_PATH_RE.match(file_name):
        if base in by_name:
            cands = by_name[base]
            cands.sort(key=lambda q: len(str(q)))
            return cands[0]

    raise FileNotFoundError(f"Could not resolve image '{file_name}' under {images_root}")


def build_resolved_static_dataset(src_dir: Path, dst_dir: Path) -> Path:
    """
    Create a lightweight copy of the static OVR dataset where all `file_name`
    entries are replaced by absolute paths on the mounted drive.

    No images are copied. RFDETR will see absolute paths and ignore its root.
    """
    ok_marker = dst_dir / ".RESOLVED_OK"
    if ok_marker.exists():
        print(f"[RESOLVE] Using cached resolved dataset: {dst_dir}")
        return dst_dir

    if not IMAGES_FALLBACK_ROOT.exists():
        raise FileNotFoundError(
            f"IMAGES_FALLBACK_ROOT does not exist: {IMAGES_FALLBACK_ROOT}\n"
            f"Set it to your mounted CellScanData root."
        )

    print(f"[RESOLVE] Building resolved dataset for {src_dir.name} → {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    # index all images once
    by_rel, by_name = _index_image_paths(IMAGES_FALLBACK_ROOT)

    for split in ("train", "valid", "test"):
        src_json = src_dir / split / "_annotations.coco.json"
        if not src_json.exists():
            continue

        data = json.loads(src_json.read_text(encoding="utf-8"))
        images = data.get("images", [])
        anns   = data.get("annotations", [])
        cats   = data.get("categories", [])

        out_images = []
        missing = 0

        for im in images:
            try:
                resolved = _resolve_image_path(im["file_name"], IMAGES_FALLBACK_ROOT, by_rel, by_name)
            except FileNotFoundError as e:
                missing += 1
                print(f"[RESOLVE][WARN] {e}")
                continue

            im2 = dict(im)
            # absolute path to mounted drive
            im2["file_name"] = str(resolved.resolve())
            out_images.append(im2)

        valid_ids = {im["id"] for im in out_images}
        out_anns = [a for a in anns if a["image_id"] in valid_ids]

        out_split_dir = dst_dir / split
        out_split_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_split_dir / "_annotations.coco.json"
        out_json.write_text(
            json.dumps({"images": out_images,
                        "annotations": out_anns,
                        "categories": cats},
                       ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"[RESOLVE] {split}: kept {len(out_images)} images, "
              f"{len(out_anns)} anns (missing={missing})")

    ok_marker.write_text("ok", encoding="utf-8")
    return dst_dir

# ───────────────────────────────────────────────
# FRACTIONAL TRAIN SPLIT (for cost curves)
# ───────────────────────────────────────────────
def build_fractional_train_split(resolved_root: Path,
                                 frac: float,
                                 cache_root: Path,
                                 seed: int = 42) -> Path:
    """
    Create a dataset copy where only a fraction of TRAIN images are kept.
    'valid' and 'test' remain unchanged.
    """
    if frac >= 0.999:
        return resolved_root  # use full data

    frac_tag = f"trainfrac_{int(frac*100):02d}"
    dst_root = cache_root / f"{resolved_root.name}_{frac_tag}"
    ok_marker = dst_root / ".FRACTION_OK"

    if ok_marker.exists():
        print(f"[FRACTION] Using cached {frac_tag} dataset: {dst_root}")
        return dst_root

    print(f"[FRACTION] Building {frac_tag} dataset under {dst_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    # copy val + test JSONs as-is
    for split in ("valid", "test"):
        src = resolved_root / split / "_annotations.coco.json"
        if not src.exists():
            continue
        out_dir = dst_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "_annotations.coco.json").write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8"
        )

    # subsample TRAIN
    src_train = resolved_root / "train" / "_annotations.coco.json"
    if not src_train.exists():
        raise FileNotFoundError(f"No train split found at {src_train}")

    data = json.loads(src_train.read_text(encoding="utf-8"))
    images = data.get("images", [])
    anns   = data.get("annotations", [])
    cats   = data.get("categories", [])

    rng = random.Random(seed)
    n_keep = max(1, int(round(len(images) * frac)))
    indices = list(range(len(images)))
    rng.shuffle(indices)
    keep_idx = set(indices[:n_keep])

    kept_images = [im for i, im in enumerate(images) if i in keep_idx]
    kept_ids    = {im["id"] for im in kept_images}
    kept_anns   = [a for a in anns if a["image_id"] in kept_ids]

    out_train_dir = dst_root / "train"
    out_train_dir.mkdir(parents=True, exist_ok=True)
    out_train = out_train_dir / "_annotations.coco.json"
    out_train.write_text(
        json.dumps({
            "images": kept_images,
            "annotations": kept_anns,
            "categories": cats,
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[FRACTION] train: kept {len(kept_images)}/{len(images)} images "
          f"({frac:.2f}), anns={len(kept_anns)}")

    ok_marker.write_text("ok", encoding="utf-8")
    return dst_root

# ───────────────────────────────────────────────
# PATCHIFY DATASET (640×640 → 224×224 crops, no resizing)
# ───────────────────────────────────────────────

def _compute_positions(length: int, patch: int, stride: int):
    """Return sorted start positions so we cover whole axis with last patch anchored at end."""
    positions = []
    x = 0
    while x + patch <= length:
        positions.append(x)
        x += stride
    last = length - patch
    if last >= 0 and (not positions or positions[-1] != last):
        positions.append(last)
    return sorted(set(positions))


def _patchify_split(
    src_json: Path,
    dst_json: Path,
    split_images_dir: Path,
    patch_size: int,
    stride: int,
    start_img_id: int,
    start_ann_id: int,
) -> tuple[int, int, int, int]:
    """
    Build a patchified COCO split:
      - loads src_json (already resolved with absolute file_name)
      - crops each image into overlapping patch_size×patch_size tiles
      - recomputes bboxes for each patch (no resizing)
    Returns (next_img_id, next_ann_id, n_new_images, n_new_anns).
    """
    data = json.loads(src_json.read_text(encoding="utf-8"))
    images = data.get("images", [])
    anns   = data.get("annotations", [])
    cats   = data.get("categories", [])

    anns_by_img = ddict(list)
    for a in anns:
        anns_by_img[a["image_id"]].append(a)

    new_images = []
    new_anns = []

    img_id = start_img_id
    ann_id = start_ann_id

    split_images_dir.mkdir(parents=True, exist_ok=True)

    for im in images:
        fname = im["file_name"]
        img_path = Path(fname)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[PATCHIFY][WARN] Failed to open {img_path}: {e}")
            continue

        W, H = img.size
        xs = _compute_positions(W, patch_size, stride)
        ys = _compute_positions(H, patch_size, stride)

        img_anns = anns_by_img.get(im["id"], [])

        stem = Path(fname).stem

        for yi, y0 in enumerate(ys):
            for xi, x0 in enumerate(xs):
                x1 = x0 + patch_size
                y1 = y0 + patch_size
                box = (x0, y0, x1, y1)

                patch = img.crop(box)
                out_name = f"{stem}_y{yi:02d}_x{xi:02d}.png"
                out_path = split_images_dir / out_name
                patch.save(out_path)

                this_img_id = img_id
                img_id += 1

                new_images.append({
                    "id": this_img_id,
                    "file_name": str(out_path),
                    "width": patch_size,
                    "height": patch_size,
                    # optional: track original
                    "original_image_id": im["id"],
                })

                # re-map bboxes
                for a in img_anns:
                    x, y, w, h = a["bbox"]
                    x2 = x + w
                    y2 = y + h

                    # intersection with patch
                    ix0 = max(x0, x)
                    iy0 = max(y0, y)
                    ix1 = min(x1, x2)
                    iy1 = min(y1, y2)

                    if ix1 <= ix0 or iy1 <= iy0:
                        continue  # no overlap

                    inter_w = ix1 - ix0
                    inter_h = iy1 - iy0
                    inter_area = inter_w * inter_h
                    orig_area = w * h if w > 0 and h > 0 else 1.0

                    # keep only if sufficient overlap (e.g. at least 25% of original box)
                    if inter_area / orig_area < 0.25:
                        continue

                    # coords relative to patch
                    nx = ix0 - x0
                    ny = iy0 - y0
                    nw = inter_w
                    nh = inter_h

                    if nw <= 1 or nh <= 1:
                        continue

                    na = dict(a)
                    na["id"] = ann_id
                    na["image_id"] = this_img_id
                    na["bbox"] = [float(nx), float(ny), float(nw), float(nh)]
                    na["area"] = float(nw * nh)
                    ann_id += 1
                    new_anns.append(na)

    out = {
        "images": new_images,
        "annotations": new_anns,
        "categories": cats,
    }
    dst_json.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    print(f"[PATCHIFY] {src_json.parent.name}: {len(new_images)} images, {len(new_anns)} anns → {dst_json}")
    return img_id, ann_id, len(new_images), len(new_anns)


def build_patchified_dataset(resolved_root: Path,
                             cache_root: Path,
                             patch_size: int = 224,
                             stride: int | None = None) -> Path:
    """
    Build a patchified COCO dataset (224×224 crops, no resizing) from a
    *resolved* dataset (file_name = absolute paths).
    """
    if stride is None:
        stride = patch_size  # can change if you want more overlap

    tag = f"PATCH{patch_size}"
    dst_root = cache_root / f"{resolved_root.name}_{tag}"
    ok_marker = dst_root / ".PATCH_OK"

    if ok_marker.exists():
        print(f"[PATCHIFY] Using cached patchified dataset: {dst_root}")
        return dst_root

    print(f"[PATCHIFY] Building patchified dataset {tag} → {dst_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    next_img_id = 1
    next_ann_id = 1

    total_imgs = 0
    total_anns = 0

    for split in ("train", "valid", "test"):
        src_json = resolved_root / split / "_annotations.coco.json"
        if not src_json.exists():
            continue

        split_dir = dst_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_images_dir = split_dir / "images"

        dst_json = split_dir / "_annotations.coco.json"

        next_img_id, next_ann_id, n_imgs, n_anns = _patchify_split(
            src_json,
            dst_json,
            split_images_dir,
            patch_size=patch_size,
            stride=stride,
            start_img_id=next_img_id,
            start_ann_id=next_ann_id,
        )
        total_imgs += n_imgs
        total_anns += n_anns

    print(f"[PATCHIFY] TOTAL: {total_imgs} images, {total_anns} annotations in {dst_root}")
    ok_marker.write_text("ok", encoding="utf-8")
    return dst_root

# ───────────────────────────────────────────────
# Auto-detect GPU count and set MAX_PARALLEL
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
# ───────────────────────────────────────────────────────────────────────────────
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
    DATASET_EPI = _autofind_dataset(DEFAULT_ROOT, "SquamousEpithelialCell_OVR")

# Where to put all outputs (HPO runs + leaderboards + final selections)
OUTPUT_ROOT = env_path("OUTPUT_ROOT", WORK_ROOT / "RFDETR_SOLO_OUTPUT" / "HPO_BOTH_OVR")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Will be set in main() only (avoid workers creating their own sessions)
HPO_SESSION_ID: str | None = None
SESSION_ROOT: Path | None = None

NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))
SEED        = int(os.getenv("SEED", "42"))

# ───────────────────────────────────────────────────────────────────────────────
# Offline augmentation / resolving / patchifying
# ───────────────────────────────────────────────────────────────────────────────
def get_or_build_aug_cache(target_name: str, dataset_dir: Path, root_out: Path, aug_copies: int) -> Path:
    """
    We:
      1) build a 'resolved' dataset where file_name paths are absolute
      2) if USE_PATCH_224, build a 224×224 patchified dataset on top of that
    """
    # step 1: resolved dataset
    resolved_cache = root_out / f"{dataset_dir.name}_RESOLVED"
    resolved_dir = build_resolved_static_dataset(dataset_dir, resolved_cache)

    if not USE_PATCH_224:
        return resolved_dir

    # step 2: patchified dataset (no resizing)
    patch_dir = build_patchified_dataset(
        resolved_root=resolved_dir,
        cache_root=root_out,
        patch_size=PATCH_SIZE,
        stride=PATCH_SIZE,   # change if you want more overlap
    )
    return patch_dir

# ───────────────────────────────────────────────────────────────────────────────
# Training (single run). Called in worker processes.
# ───────────────────────────────────────────────────────────────────────────────
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
                  backbone_ckpt: str | None = None,
                  train_fraction: float = 1.0,
                  seed: int = 42,
                  num_workers: int = 8,
                  pin_memory: bool = True,
                  persistent_workers: bool = True) -> dict:

    # Build resolved / optional patchified dataset
    data_dir = get_or_build_aug_cache(target_name, dataset_dir, OUTPUT_ROOT, aug_copies)

    # Optionally reduce the TRAIN split size
    if train_fraction < 0.999:
        data_dir = build_fractional_train_split(
            resolved_root=data_dir,
            frac=train_fraction,
            cache_root=OUTPUT_ROOT,
            seed=seed,
        )

    # Force resolution from selected input mode for consistent runs.
    if USE_PATCH_224:
        resolution = PATCH_SIZE
    else:
        resolution = FULL_RESOLUTION

    model = model_cls()
    sig = signature(model.train)
    can = set(sig.parameters.keys())

    # ---- base kwargs ----
    kwargs = dict(
        dataset_dir=str(data_dir),
        output_dir=str(out_dir),
        class_names=[target_name],

        resolution=resolution,
        batch_size=batch_size,
        grad_accum_steps=8,
        epochs=epochs,
        lr=lr,
        weight_decay=5e-4,
        dropout=0.1,
        num_queries=num_queries,

        multi_scale=False,
        gradient_checkpointing=True,
        amp=True,
        num_workers=min(num_workers, os.cpu_count() or num_workers),
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,

        seed=seed,
        early_stopping=True,
        checkpoint_interval=10,
        run_test=True,
    )

    def maybe(name, value):
        if name in can and value is not None:
            kwargs[name] = value

    # Backbone / encoder checkpoint from SSL
    maybe("encoder_name", backbone_ckpt)
    maybe("pretrained_backbone", backbone_ckpt)

    # resizing behaviour: IMPORTANT
    if USE_PATCH_224:
        # patches are already square 224×224; do NOT resize
        maybe("square_resize_div_64", False)
        maybe("do_random_resize_via_padding", False)
        maybe("random_resize_via_padding", False)
    else:
        # previous behaviour
        maybe("square_resize_div_64", True)
        maybe("do_random_resize_via_padding", False)
        maybe("random_resize_via_padding", False)

    # Log meta
    meta = out_dir / "run_meta"
    meta.mkdir(parents=True, exist_ok=True)
    kwargs["train_fraction"] = float(train_fraction)
    kwargs["input_mode"] = INPUT_MODE
    kwargs["use_patch_224"] = USE_PATCH_224
    kwargs["patch_size"] = PATCH_SIZE if USE_PATCH_224 else None
    kwargs["full_resolution"] = None if USE_PATCH_224 else FULL_RESOLUTION
    (meta / "train_kwargs.json").write_text(json.dumps(kwargs, indent=2), encoding="utf-8")
    (meta / "dataset_info.json").write_text(json.dumps({
        "dataset_dir_original": str(dataset_dir),
        "dataset_dir_effective": str(data_dir),
        "target_name": target_name
    }, indent=2), encoding="utf-8")

    print(f"[TRAIN] {model.__class__.__name__} — {target_name} → {out_dir}")
    model.train(**kwargs)

    best = find_best_val(out_dir)
    (out_dir / "val_best_summary.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best

# ───────────────────────────────────────────────────────────────────────────────
def find_best_val(output_dir: Path) -> dict:
    # 1) prefer RFDETR results.json if present
    p = output_dir / "results.json"
    if p.exists():
        js = json.loads(p.read_text())
        valid = js.get("class_map", {}).get("valid", [])
        val_row = None
        for r in valid:
            if r.get("class") == "all":
                val_row = r
                break
        if val_row is None and valid:
            val_row = valid[0]
        if val_row is not None:
            return {
                "best_epoch": None,  # RFDETR doesn't log epoch in results.json
                "map50": float(val_row.get("map@50", 0.0)),
                "map5095": float(val_row.get("map@50:95", 0.0)),
                "source": "results.json",
            }

    # 2) fall back to old behaviour (also check eval/ subfolder)
    candidates = [
        "val_best_summary.json",
        "val_metrics.json", "metrics_val.json", "coco_eval_val.json",
        "metrics.json", "val_results.json", "results_val.json",
    ]
    for name in candidates:
        for base in (output_dir, output_dir / "eval"):
            p = base / name
            if not p.exists():
                continue
            try:
                js = json.loads(p.read_text(encoding="utf-8"))
                def pick(keys):
                    for k in keys:
                        if k in js:
                            return float(js[k])
                return {
                    "best_epoch": js.get("best") or js.get("best_epoch") or js.get("epoch"),
                    "map50":      pick(["map50","mAP50","ap50","AP50","bbox/AP50"]),
                    "map5095":    pick(["map","mAP","mAP5095","bbox/mAP"]),
                    "source":     name,
                }
            except Exception:
                continue
    return {"best_epoch": None, "map50": None, "map5095": None, "source": "not_found"}

# ───────────────────────────────────────────────────────────────────────────────
# HPO spaces (compact)
# ───────────────────────────────────────────────────────────────────────────────
SEARCH_RESOLUTION = PATCH_SIZE if USE_PATCH_224 else FULL_RESOLUTION

SEARCH_LEUCO = {
    # Model & input
    "MODEL_CLS":       ["RFDETRLarge"],
    "RESOLUTION":      [SEARCH_RESOLUTION],  # mode-fixed (224 patch or full resolution)
    "EPOCHS":          [80],

    # Optimizer / schedule
    "LR":              [5e-5, 8e-5],
    "LR_ENCODER_MULT": [1.0],
    "BATCH":           [4],
    "WARMUP_STEPS":    [0, 4000],

    # Detector capacity
    "NUM_QUERIES":     [384, 512],

    # Aug / regularization (we’re using RFDETR defaults; these are just logged)
    "AUG_COPIES":      [0],
    "SCALE_RANGE":     [(0.9, 1.1)],
    "ROT_DEG":         [5.0],
    "COLOR_JITTER":    [0.20],
    "GAUSS_BLUR":      [0.10],
}
'''
SEARCH_EPI = {
    "MODEL_CLS":       ["RFDETRLarge"],
    "RESOLUTION":      [672],      # ignored in patch mode (forced to 224)
    "EPOCHS":          [80],

    "LR":              [5e-5],
    "LR_ENCODER_MULT": [1.0],
    "BATCH":           [8],
    "WARMUP_STEPS":    [0],

    "NUM_QUERIES":     [250],

    "AUG_COPIES":      [0],
    "SCALE_RANGE":     [(0.9, 1.1)],
    "ROT_DEG":         [5.0],
    "COLOR_JITTER":    [0.20],
    "GAUSS_BLUR":      [0.20],

    # Example: fixed best SSL backbone; uncomment when you want SSL
    # "ENCODER_CKPT":    [BEST_SSL_CKPT],

    # Data-cost curve
    "TRAIN_FRACTION":  [0.25, 0.50, 0.75, 1.00],
}'''

# Best supervised RFDETR HPO space for 224×224 patches (EPI)
#224
SEARCH_EPI = {
    # Model & input
    "MODEL_CLS":       ["RFDETRLarge"],   # keep Large, patches are smaller so this is fine
    "RESOLUTION":      [SEARCH_RESOLUTION],  # mode-fixed (224 patch or full resolution)
    "EPOCHS":          [80],              # let early stopping decide

    # Optimizer / schedule
    "LR":              [5e-5],  # a bit wider around your previous good LR
    "LR_ENCODER_MULT": [1.0],              # still ignored for now
    "BATCH":           [16],            # 16 should be feasible with 224×224, OOM-backoff will save us
    "WARMUP_STEPS":    [0],          # no warmup vs modest warmup

    # Detector capacity — per 224×224 patch there are few cells
    "NUM_QUERIES":     [200],         # plenty for “few objects per patch”

    # Augmentation / regularization
    "AUG_COPIES":      [0],                # still no offline aug (RFDETR has its own online aug)
    "SCALE_RANGE":     [(0.9, 1.1)],
    "ROT_DEG":         [5.0],
    "COLOR_JITTER":    [0.20],
    "GAUSS_BLUR":      [0.20],

    # Supervised baseline only → NO SSL backbone here
    "ENCODER_CKPT":  [BEST_SSL_CKPT],               # intentionally omitted

    # For this HPO: always use full train split
    "TRAIN_FRACTION":  [0.125, 0.03, 1],
}


def grid(space: dict):
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

# ───────────────────────────────────────────────────────────────────────────────
# Trial-level parallel launcher with allocator + OOM backoff
# ───────────────────────────────────────────────────────────────────────────────
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
        mc = cfg["MODEL_CLS"]
        if isinstance(mc, str):
            model_cls = name2cls[mc]
        else:
            model_cls = name2cls.get(getattr(mc, "__name__", "RFDETRLarge"), RFDETRLarge)

        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        cur = torch.cuda.current_device()
        print(f"[WORKER {run_idx:03d}] Physical GPU {gpu_id} masked as cuda:0 | "
              f"CUDA_VISIBLE_DEVICES={vis} | using cuda:{cur} ({torch.cuda.get_device_name(cur)})")

        _set_perf_toggles()

        out_root = Path(out_root)
        run_dir = out_root / f"HPO_Config_{run_idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        backbone_ckpt = cfg.get("ENCODER_CKPT")
        train_fraction = float(cfg.get("TRAIN_FRACTION", 1.0))

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
                backbone_ckpt=backbone_ckpt,
                train_fraction=train_fraction,
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
                    backbone_ckpt=backbone_ckpt,
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
            "input_mode": INPUT_MODE,
            "use_patch_224": USE_PATCH_224,
            "patch_size": PATCH_SIZE if USE_PATCH_224 else None,
            "full_resolution": None if USE_PATCH_224 else FULL_RESOLUTION,
        }
        (run_dir / "hpo_record.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        result_q.put(("ok", row))
    except Exception as e:
        result_q.put(("err", {"run_idx": run_idx, "target": target_name, "error": repr(e)}))

# ───────────────────────────────────────────────────────────────────────────────
def run_hpo_all_classes(session_root: Path) -> dict:
    """
    Multi-class HPO launcher.
    - Uses ALL visible GPUs as a shared pool.
    - Each run occupies ONE GPU until it finishes.
    - Classes are selected by HPO_TARGET ('leu', 'epi', 'all').
    """
    selected = []

    if HPO_TARGET in ("leu", "all"):
        selected.append(("Leucocyte", DATASET_LEUCO, SEARCH_LEUCO))
    if HPO_TARGET in ("epi", "all"):
        selected.append(("Squamous Epithelial Cell", DATASET_EPI, SEARCH_EPI))

    if not selected:
        raise RuntimeError(f"HPO_TARGET={HPO_TARGET!r} did not match any classes (expected 'leu', 'epi', or 'all').")

    class_out_roots = {}
    for target_name, _, _ in selected:
        class_token = target_name.replace(" ", "")
        out_root = session_root / class_token
        out_root.mkdir(parents=True, exist_ok=True)
        class_out_roots[target_name] = out_root

    gpu_ids = _get_visible_gpu_ids()
    if not gpu_ids:
        raise RuntimeError("No visible GPUs. Set CUDA_VISIBLE_DEVICES or ensure CUDA is available.")
    max_parallel = max(1, min(MAX_PARALLEL, len(gpu_ids)))
    print(f"[HPO] Visible GPUs: {gpu_ids}  |  MAX_PARALLEL={max_parallel}")
    print(f"[HPO] Input mode: {INPUT_MODE} (patch={USE_PATCH_224}, full_resolution={FULL_RESOLUTION})")

    jobs = []
    for target_name, dataset_dir, search_space in selected:
        for cfg in grid(search_space):
            jobs.append({
                "target_name": target_name,
                "dataset_dir": str(dataset_dir),
                "out_root": str(class_out_roots[target_name]),
                "cfg": cfg,
            })

    print(f"[HPO] Total jobs across classes={len(jobs)}")

    ctx = mp.get_context("spawn")
    result_q: mp.Queue = ctx.Queue()
    active: dict[int, mp.Process] = {}
    leaderboard_all = []
    gpu_count = len(gpu_ids)
    next_gpu_slot = 0
    next_run_idx = 0
    job_iter = iter(jobs)

    def launch_job(run_idx: int, job: dict):
        nonlocal next_gpu_slot
        gpu_id = gpu_ids[next_gpu_slot]
        next_gpu_slot = (next_gpu_slot + 1) % gpu_count

        p = ctx.Process(
            target=_worker_entry,
            args=(
                job["cfg"],
                gpu_id,
                run_idx,
                job["target_name"],
                job["dataset_dir"],
                job["out_root"],
                result_q,
            ),
        )
        p.daemon = False
        p.start()
        active[run_idx] = p
        print(f"[HPO] Launched run {run_idx:03d} on GPU {gpu_id} "
              f"for target='{job['target_name']}' with cfg={job['cfg']}")

    def _drain_results_blocking():
        nonlocal leaderboard_all
        while not result_q.empty():
            status, payload = result_q.get()
            rid = payload.get("run_idx")
            if rid in active and not active[rid].is_alive():
                active[rid].join()
                del active[rid]
            if status == "ok":
                leaderboard_all.append(payload)
                print(f"[HPO] Finished run {payload['run_idx']:03d} "
                      f"({payload['target']}) — AP50={payload['val_AP50']} mAP={payload['val_mAP5095']}")
            else:
                print(f"[HPO] ERROR run {payload.get('run_idx')} "
                      f"({payload.get('target')}): {payload.get('error')}")

    try:
        while True:
            while len([p for p in active.values() if p.is_alive()]) < max_parallel:
                try:
                    job = next(job_iter)
                except StopIteration:
                    break
                next_run_idx += 1
                launch_job(next_run_idx, job)

            if not active:
                break

            _drain_results_blocking()
            time.sleep(1)

    except StopIteration:
        pass

    while active:
        _drain_results_blocking()
        time.sleep(1)

    if not leaderboard_all:
        print("[HPO] No successful runs.")
        return {}

    per_class_rows = defaultdict(list)
    for row in leaderboard_all:
        per_class_rows[row["target"]].append(row)

    results = {}

    for target_name, rows in per_class_rows.items():
        class_token = target_name.replace(" ", "")
        out_root = class_out_roots[target_name]

        def sort_key(r):
            a = r["val_AP50"]; b = r["val_mAP5095"]
            return (-(a if a is not None else -1), -(b if b is not None else -1))

        rows.sort(key=sort_key)

        csv_path = out_root / "hpo_leaderboard.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)

        (out_root / "hpo_top.json").write_text(json.dumps(rows[:5], indent=2), encoding="utf-8")
        best_row = rows[0]
        (out_root / "hpo_best.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")

        results[target_name] = {
            "leaderboard": rows,
            "best": best_row,
            "out_dir": str(out_root),
        }

    return results

# ───────────────────────────────────────────────────────────────────────────────
def main():
    global HPO_SESSION_ID, SESSION_ROOT

    print("[WORK_ROOT]", WORK_ROOT)
    print("[DATASETS]")
    print("  Leucocyte:", DATASET_LEUCO)
    print("  Epithelial:", DATASET_EPI)
    for p in (DATASET_LEUCO, DATASET_EPI):
        for part in ("train", "valid"):
            if not (p / part / "_annotations.coco.json").exists():
                raise FileNotFoundError(f"Missing {part} split in {p}")

    # Create a single timestamped session directory for THIS script run
    HPO_SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    SESSION_ROOT = OUTPUT_ROOT / f"session_{HPO_SESSION_ID}"
    SESSION_ROOT.mkdir(parents=True, exist_ok=False)
    print(f"[HPO] Session root: {SESSION_ROOT}")

    res_all = run_hpo_all_classes(SESSION_ROOT)

    final = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": HPO_SESSION_ID,
        "work_root": str(WORK_ROOT),
        "input_mode": INPUT_MODE,
        "use_patch_224": USE_PATCH_224,
        "patch_size": PATCH_SIZE if USE_PATCH_224 else None,
        "full_resolution": None if USE_PATCH_224 else FULL_RESOLUTION,
    }

    if "Leucocyte" in res_all:
        final["leucocyte"] = {
            "dataset": str(DATASET_LEUCO),
            "best": res_all["Leucocyte"]["best"],
            "leaderboard_top5": res_all["Leucocyte"]["leaderboard"][:5],
        }

    if "Squamous Epithelial Cell" in res_all:
        final["epithelial"] = {
            "dataset": str(DATASET_EPI),
            "best": res_all["Squamous Epithelial Cell"]["best"],
            "leaderboard_top5": res_all["Squamous Epithelial Cell"]["leaderboard"][:5],
        }

    summary_path = SESSION_ROOT / "FINAL_HPO_SUMMARY.json"
    summary_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("\n[FINAL] Summary →", summary_path)
    print(json.dumps(final, indent=2))

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
