# ===========================
# FILE: Best_SSL_Model_Selection.py
# ===========================
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from inspect import signature
import os, sys, json, glob, os.path as op, time, csv, re, random, subprocess
from collections import defaultdict as ddict
from PIL import Image


# ───────────────────────────────────────────────────────────────────────────────
# TOGGLES
# ───────────────────────────────────────────────────────────────────────────────
# Which classes to run:
#  - "leu"  -> only Leucocyte
#  - "epi"  -> only Squamous Epithelial Cell
#  - "all"  -> both
PROBE_TARGET = os.environ.get("RFDETR_PROBE_TARGET", "epi").lower()

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
        full_resolution = 640
        return use_patch, str(mode_size), patch_size, full_resolution

    # Backward-compatible path: keep old env behavior if RFDETR_INPUT_MODE is unset.
    use_patch = bool(int(os.getenv("RFDETR_USE_PATCH_224", "1")))
    patch_size = int(os.getenv("RFDETR_PATCH_SIZE", "224"))
    full_resolution = int(os.getenv("RFDETR_FULL_RESOLUTION", "640"))
    return use_patch, (str(patch_size) if use_patch else "640"), patch_size, full_resolution


# If RFDETR_INPUT_MODE != 640 we run patch mode with patch size = int(RFDETR_INPUT_MODE).
# If RFDETR_INPUT_MODE == 640 we run full-image mode.
USE_PATCH_224, INPUT_MODE, PATCH_SIZE, FULL_RESOLUTION = _resolve_input_mode()

# Fraction of TRAIN split to use for *all* runs (must be same across ckpts)
TRAIN_FRACTION = float(os.getenv("RFDETR_TRAIN_FRACTION", "0.125"))
FRACTION_SEED  = int(os.getenv("RFDETR_FRACTION_SEED", "42"))

# Static training seed for RFDETR
SEED = int(os.getenv("SEED", "42"))
PARALLEL_GPUS = int(os.getenv("RFDETR_PARALLEL_GPUS", "1"))

print(f"[PROBE] Target classes: {PROBE_TARGET!r} (env RFDETR_PROBE_TARGET)")
print(f"[INPUT MODE] RFDETR_INPUT_MODE={INPUT_MODE}  USE_PATCH_224={USE_PATCH_224}")
print(f"[INPUT SIZE] PATCH_SIZE={PATCH_SIZE}  FULL_RESOLUTION={FULL_RESOLUTION}")
if os.getenv("RFDETR_INPUT_MODE", "").strip():
    print("[INPUT MODE] RFDETR_INPUT_MODE is authoritative; legacy RFDETR_USE_PATCH_224/RFDETR_PATCH_SIZE are ignored.")
print(f"[PROBE] TRAIN_FRACTION={TRAIN_FRACTION}  FRACTION_SEED={FRACTION_SEED}  SEED={SEED}")
print(f"[PROBE] RFDETR_PARALLEL_GPUS={PARALLEL_GPUS}")

if not (0.0 < TRAIN_FRACTION <= 1.0):
    raise ValueError(f"RFDETR_TRAIN_FRACTION must be in (0, 1], got {TRAIN_FRACTION}")
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


def env_str(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


# Where the **real images** live (on your drive) for resolving COCO file_name paths
IMAGES_FALLBACK_ROOT = env_path(
    "IMAGES_FALLBACK_ROOT",
    WORK_ROOT / "CellScanData" / "Zoom10x - Quality Assessment_Cleaned",
)

# ───────────────────────────────────────────────────────────────────────────────
# OUTPUT ROOT (selection runs)
#   - One timestamped session per script run
#   - OUTPUT_ROOT is an alias used for caches + run outputs
# ───────────────────────────────────────────────────────────────────────────────
OUTPUT_BASE = env_path("OUTPUT_BASE", WORK_ROOT / "Linear_Probing_For_SSL" / "SSL_SELECTION")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

SESSION_ROOT_ENV = os.getenv("RFDETR_SESSION_ROOT", "").strip()
if SESSION_ROOT_ENV:
    SESSION_ROOT = Path(SESSION_ROOT_ENV)
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    SESSION_ID = os.getenv("RFDETR_SESSION_ID", "").strip() or SESSION_ROOT.name.replace("session_", "")
else:
    SESSION_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
    SESSION_ROOT = OUTPUT_BASE / f"session_{SESSION_ID}"
    SESSION_ROOT.mkdir(parents=True, exist_ok=False)

OUTPUT_ROOT = SESSION_ROOT  # compatibility alias

print(f"[PROBE] OUTPUT_BASE : {OUTPUT_BASE}")
print(f"[PROBE] SESSION_ROOT: {SESSION_ROOT}")

# ───────────────────────────────────────────────────────────────────────────────
# SSL CHECKPOINT SWEEP
# ───────────────────────────────────────────────────────────────────────────────
SSL_CKPT_ROOT = env_path("SSL_CKPT_ROOT", WORK_ROOT / "SSL_Checkpoints")
if not SSL_CKPT_ROOT.exists():
    fallback_ckpt_root = WORK_ROOT / "Checkpoints"
    if fallback_ckpt_root.exists():
        print(f"[WARN] SSL_CKPT_ROOT does not exist: {SSL_CKPT_ROOT}")
        print(f"[WARN] Falling back to: {fallback_ckpt_root}")
        SSL_CKPT_ROOT = fallback_ckpt_root
    else:
        raise FileNotFoundError(
            f"SSL_CKPT_ROOT does not exist: {SSL_CKPT_ROOT}\n"
            f"Also checked fallback: {fallback_ckpt_root}"
        )

# Option A: provide explicit list via env (comma-separated)
#   export SSL_CKPTS="epoch_epoch-004.ckpt,epoch_epoch-014.ckpt,last.ckpt"
_ssl_list = os.getenv("SSL_CKPTS", "").strip()


def _find_ssl_ckpts(root: Path) -> list[Path]:
    cands: list[Path] = []
    for ext in ("*.ckpt", "*.pth", "*.pt"):
        cands += list(root.glob(ext))

    if not cands:
        raise FileNotFoundError(f"No SSL checkpoints found in {root}")

    def sort_key(p: Path):
        name = p.name.lower()
        if "last" in name:
            return (10**9, name)
        m = re.search(r"(epoch|ep)[^\d]*(\d+)", name)
        if m:
            return (int(m.group(2)), name)
        m2 = re.search(r"(\d+)", name)
        if m2:
            return (int(m2.group(1)), name)
        return (10**8, name)

    cands.sort(key=sort_key)
    return cands


if _ssl_list:
    SSL_BACKBONES = [SSL_CKPT_ROOT / s.strip() for s in _ssl_list.split(",") if s.strip()]
else:
    SSL_BACKBONES = _find_ssl_ckpts(SSL_CKPT_ROOT)

print("[SSL BACKBONES]")
for p in SSL_BACKBONES:
    print("  ", p)

STAT_DATASETS_ROOT = env_path(
    "STAT_DATASETS_ROOT",
    Path("/work/projects/myproj/SOLO_Supervised_RFDETR/Stat_Dataset"),
)
DATASET_PREFIX_LEU = env_str("DATASET_PREFIX_LEU", "QA-2025v2_Leucocyte_OVR")
DATASET_PREFIX_EPI = env_str("DATASET_PREFIX_EPI", "QA-2025v2_SquamousEpithelialCell_OVR")
DATASET_LEUCO_DIR_OVERRIDE = os.getenv("DATASET_LEUCO_DIR", "").strip()
DATASET_EPI_DIR_OVERRIDE = os.getenv("DATASET_EPI_DIR", "").strip()

# ───────────────────────────────────────────────
# Path resolution helpers for COCO
# ───────────────────────────────────────────────
VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
WINDOWS_PATH_RE = re.compile(r"^[a-zA-Z]:\\")  # detect old Windows-style paths


def _index_image_paths(root: Path):
    by_rel = {}
    by_name = ddict(list)
    root = root.resolve()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            rel = p.resolve().relative_to(root).as_posix()
            by_rel[rel] = p
            by_name[p.name].append(p)
    return by_rel, by_name


def _resolve_image_path(file_name: str, images_root: Path, by_rel: dict, by_name: dict) -> Path:
    rel = file_name.replace("\\", "/")
    direct = (images_root / rel)
    if direct.exists():
        return direct
    if rel in by_rel:
        return by_rel[rel]
    base = Path(rel).name
    if base in by_name:
        cands = by_name[base]
        if len(cands) == 1:
            return cands[0]
        cands.sort(key=lambda q: len(str(q)))
        return cands[0]
    if WINDOWS_PATH_RE.match(file_name):
        if base in by_name:
            cands = by_name[base]
            cands.sort(key=lambda q: len(str(q)))
            return cands[0]
    raise FileNotFoundError(f"Could not resolve image '{file_name}' under {images_root}")


def build_resolved_static_dataset(src_dir: Path, dst_dir: Path) -> Path:
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
            im2["file_name"] = str(resolved.resolve())
            out_images.append(im2)

        valid_ids = {im["id"] for im in out_images}
        out_anns = [a for a in anns if a["image_id"] in valid_ids]

        out_split_dir = dst_dir / split
        out_split_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_split_dir / "_annotations.coco.json"
        out_json.write_text(
            json.dumps({"images": out_images, "annotations": out_anns, "categories": cats}, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[RESOLVE] {split}: kept {len(out_images)} images, {len(out_anns)} anns (missing={missing})")

    ok_marker.write_text("ok", encoding="utf-8")
    return dst_dir


# ───────────────────────────────────────────────
# FRACTIONAL TRAIN SPLIT (deterministic + cached)
# ───────────────────────────────────────────────
def build_fractional_train_split(resolved_root: Path, frac: float, cache_root: Path, seed: int = 42) -> Path:
    if frac >= 0.999:
        return resolved_root

    frac_tag = f"trainfrac_{int(frac*100):02d}_seed{seed}"
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
        (out_dir / "_annotations.coco.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # subsample TRAIN deterministically
    src_train = resolved_root / "train" / "_annotations.coco.json"
    if not src_train.exists():
        raise FileNotFoundError(f"No train split found at {src_train}")

    data = json.loads(src_train.read_text(encoding="utf-8"))
    images = data.get("images", [])
    anns   = data.get("annotations", [])
    cats   = data.get("categories", [])
    if not images:
        raise ValueError(f"Train split has zero images at {src_train}")

    rng = random.Random(seed)
    n_keep = max(1, int(round(len(images) * frac)))
    indices = list(range(len(images)))
    rng.shuffle(indices)
    keep_idx = set(indices[:n_keep])

    kept_images = [im for i, im in enumerate(images) if i in keep_idx]
    kept_ids    = {im["id"] for im in kept_images}
    if len(kept_images) != len(kept_ids):
        raise ValueError(f"Duplicate image ids detected in train split: {src_train}")
    kept_anns   = [a for a in anns if a["image_id"] in kept_ids]

    out_train_dir = dst_root / "train"
    out_train_dir.mkdir(parents=True, exist_ok=True)
    out_train = out_train_dir / "_annotations.coco.json"
    out_train.write_text(
        json.dumps({"images": kept_images, "annotations": kept_anns, "categories": cats}, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[FRACTION] train: kept {len(kept_images)}/{len(images)} images ({frac:.3f}), anns={len(kept_anns)}")
    ok_marker.write_text("ok", encoding="utf-8")
    return dst_root


# ───────────────────────────────────────────────
# PATCHIFY DATASET (640×640 → 224×224 crops, no resizing)
# (copied from your HPO script)
# ───────────────────────────────────────────────
def _compute_positions(length: int, patch: int, stride: int):
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
    progress_every_images: int = 25,
) -> tuple[int, int, int, int]:
    """
    Build a patchified COCO split:
      - src_json must already have absolute 'file_name' paths
      - crops each image into patch_size×patch_size tiles with given stride
      - remaps bboxes into each patch and filters by overlap threshold
    Returns (next_img_id, next_ann_id, n_new_images, n_new_anns)
    """

    t0 = time.time()

    data = json.loads(src_json.read_text(encoding="utf-8"))
    images = data.get("images", [])
    anns   = data.get("annotations", [])
    cats   = data.get("categories", [])

    # group annotations per original image id
    anns_by_img = ddict(list)
    for a in anns:
        anns_by_img[a["image_id"]].append(a)

    new_images: list[dict] = []
    new_anns: list[dict] = []

    img_id = start_img_id
    ann_id = start_ann_id

    split_images_dir.mkdir(parents=True, exist_ok=True)

    n_src = len(images)
    split_name = src_json.parent.name

    for idx, im in enumerate(images, start=1):
        if progress_every_images and (idx == 1 or idx % progress_every_images == 0):
            print(f"[PATCHIFY] {split_name}: {idx}/{n_src} source images... (new_imgs={len(new_images)}, new_anns={len(new_anns)})")

        img_path = Path(im["file_name"])
        img_anns = anns_by_img.get(im["id"], [])
        stem = img_path.stem

        try:
            # context manager ensures file handle is closed
            with Image.open(img_path) as _img:
                img = _img.convert("RGB")
                W, H = img.size

                xs = _compute_positions(W, patch_size, stride)
                ys = _compute_positions(H, patch_size, stride)

                for yi, y0 in enumerate(ys):
                    for xi, x0 in enumerate(xs):
                        x1 = x0 + patch_size
                        y1 = y0 + patch_size

                        out_name = f"{stem}_y{yi:02d}_x{xi:02d}.png"
                        out_path = split_images_dir / out_name

                        # crop + save patch
                        patch = img.crop((x0, y0, x1, y1))
                        patch.save(out_path)

                        this_img_id = img_id
                        img_id += 1

                        new_images.append({
                            "id": this_img_id,
                            "file_name": str(out_path),
                            "width": patch_size,
                            "height": patch_size,
                            "original_image_id": im["id"],
                        })

                        # remap bboxes into patch coordinates
                        for a in img_anns:
                            x, y, w, h = a["bbox"]
                            if w <= 0 or h <= 0:
                                continue

                            x2 = x + w
                            y2 = y + h

                            ix0 = max(x0, x)
                            iy0 = max(y0, y)
                            ix1 = min(x1, x2)
                            iy1 = min(y1, y2)

                            if ix1 <= ix0 or iy1 <= iy0:
                                continue

                            inter_w = ix1 - ix0
                            inter_h = iy1 - iy0
                            inter_area = inter_w * inter_h
                            orig_area = w * h

                            # keep only if sufficient overlap of original bbox
                            if (inter_area / orig_area) < 0.25:
                                continue

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

        except Exception as e:
            print(f"[PATCHIFY][WARN] Failed to open/process {img_path}: {e}")
            continue

    out = {"images": new_images, "annotations": new_anns, "categories": cats}
    dst_json.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    dt = round(time.time() - t0, 2)
    print(f"[PATCHIFY] {split_name}: wrote {len(new_images)} patch-images, {len(new_anns)} anns → {dst_json}  ({dt}s)")

    return img_id, ann_id, len(new_images), len(new_anns)


def build_patchified_dataset(
    resolved_root: Path,
    cache_root: Path,
    patch_size: int = 224,
    stride: int | None = None,
    progress_every_images: int = 25,
) -> Path:
    """
    Build a patchified COCO dataset from a *resolved* dataset (absolute file_name paths).
    Produces:
      dst_root/{train,valid,test}/images/*.png
      dst_root/{train,valid,test}/_annotations.coco.json
    """

    if stride is None:
        stride = patch_size

    tag = f"PATCH{patch_size}_STRIDE{stride}"
    dst_root = cache_root / f"{resolved_root.name}_{tag}"
    ok_marker = dst_root / ".PATCH_OK"

    print("[PATCHIFY] Starting patch generation...")
    print(f"[PATCHIFY] resolved_root={resolved_root}")
    print(f"[PATCHIFY] dst_root={dst_root}")

    if ok_marker.exists():
        print(f"[PATCHIFY] Using cached patchified dataset: {dst_root}")
        return dst_root

    dst_root.mkdir(parents=True, exist_ok=True)

    next_img_id = 1
    next_ann_id = 1
    total_imgs = 0
    total_anns = 0

    for split in ("train", "valid", "test"):
        src_json = resolved_root / split / "_annotations.coco.json"
        if not src_json.exists():
            print(f"[PATCHIFY] {split}: skip (no _annotations.coco.json)")
            continue

        split_dir = dst_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        split_images_dir = split_dir / "images"
        dst_json = split_dir / "_annotations.coco.json"

        next_img_id, next_ann_id, n_imgs, n_anns = _patchify_split(
            src_json=src_json,
            dst_json=dst_json,
            split_images_dir=split_images_dir,
            patch_size=patch_size,
            stride=stride,
            start_img_id=next_img_id,
            start_ann_id=next_ann_id,
            progress_every_images=progress_every_images,
        )
        total_imgs += n_imgs
        total_anns += n_anns

    print(f"[PATCHIFY] TOTAL: {total_imgs} patch-images, {total_anns} annotations in {dst_root}")
    ok_marker.write_text("ok", encoding="utf-8")
    return dst_root



def get_or_build_probe_dataset(dataset_dir: Path, root_out: Path) -> Path:
    """
    Build the *single* dataset that ALL checkpoints will train on for this class:
      1) resolved dataset (absolute file_name)
      2) optional patchified dataset (no resizing)
      3) deterministic fractional TRAIN split (cached)
    """
    resolved_cache = root_out / f"{dataset_dir.name}_RESOLVED"
    resolved_dir = build_resolved_static_dataset(dataset_dir, resolved_cache)

    data_dir = resolved_dir
    if USE_PATCH_224:
        data_dir = build_patchified_dataset(
            resolved_root=resolved_dir,
            cache_root=root_out,
            patch_size=PATCH_SIZE,
            stride=PATCH_SIZE,
        )

    data_dir = build_fractional_train_split(
        resolved_root=data_dir,
        frac=TRAIN_FRACTION,
        cache_root=root_out,
        seed=FRACTION_SEED,
    )
    return data_dir


# ───────────────────────────────────────────────────────────────────────────────
# Metrics extraction (reuse your find_best_val)
# ───────────────────────────────────────────────────────────────────────────────
def find_best_val(output_dir: Path) -> dict:
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
                "best_epoch": None,
                "map50": float(val_row.get("map@50", 0.0)),
                "map5095": float(val_row.get("map@50:95", 0.0)),
                "source": "results.json",
            }

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
                        if k in js and js[k] is not None:
                            return float(js[k])

                return {
                    "best_epoch": js.get("best") or js.get("best_epoch") or js.get("epoch"),
                    "map50":      pick(["map50", "mAP50", "ap50", "AP50", "bbox/AP50"]),
                    "map5095":    pick(["map", "mAP", "mAP5095", "bbox/mAP"]),
                    "source":     name,
                }
            except Exception:
                continue
    return {"best_epoch": None, "map50": None, "map5095": None, "source": "not_found"}


# ───────────────────────────────────────────────────────────────────────────────
# STATIC RF-DETR TRAINING CONFIG (ONLY SSL CKPT CHANGES)
# ───────────────────────────────────────────────────────────────────────────────
STATIC_CFG = dict(
    MODEL_CLS="RFDETRLarge",
    RESOLUTION=PATCH_SIZE if USE_PATCH_224 else FULL_RESOLUTION,
    EPOCHS=40,
    LR=5e-5,
    LR_ENCODER_MULT=1.0,
    BATCH=16 if USE_PATCH_224 else 4,
    WARMUP_STEPS=0,
    NUM_QUERIES=200,
    AUG_COPIES=0,
    SCALE_RANGE=(0.9, 1.1),
    ROT_DEG=5.0,
    COLOR_JITTER=0.20,
    GAUSS_BLUR=0.20,
)

NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))


def _detect_visible_gpu_ids() -> list[str]:
    visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [t.strip() for t in visible.split(",") if t.strip()]
        return ids
    try:
        import torch
        n = int(torch.cuda.device_count())
        return [str(i) for i in range(n)] if n > 0 else []
    except Exception:
        return []


def _gpu_slots_for_parallel() -> list[str | None]:
    ids = _detect_visible_gpu_ids()
    if not ids:
        return [None]
    n = max(1, min(PARALLEL_GPUS, len(ids)))
    return ids[:n]


def _build_probe_row(
    *,
    idx: int,
    total: int,
    target_name: str,
    ckpt: Path,
    run_dir: Path,
    best: dict,
    dur: float,
    dataset_dir_effective: Path,
) -> dict:
    return {
        "idx": idx,
        "total": total,
        "target": target_name,
        "ssl_ckpt": str(ckpt),
        "ssl_name": ckpt.name,
        "dataset_dir_effective": str(dataset_dir_effective),
        "val_AP50": best.get("map50"),
        "val_mAP5095": best.get("map5095"),
        "metrics_source": best.get("source"),
        "output_dir": str(run_dir),
        "seconds": dur,
        "train_fraction": TRAIN_FRACTION,
        "fraction_seed": FRACTION_SEED,
        "seed": SEED,
        "input_mode": INPUT_MODE,
        "use_patch_224": USE_PATCH_224,
        "patch_size": PATCH_SIZE if USE_PATCH_224 else None,
        "full_resolution": None if USE_PATCH_224 else FULL_RESOLUTION,
    }


def train_one_run(target_name: str, dataset_dir_effective: Path, out_dir: Path, backbone_ckpt: str) -> dict:
    from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge
    name2cls = {"RFDETRSmall": RFDETRSmall, "RFDETRMedium": RFDETRMedium, "RFDETRLarge": RFDETRLarge}
    model_cls = name2cls[STATIC_CFG["MODEL_CLS"]]

    model = model_cls()
    sig = signature(model.train)
    can = set(sig.parameters.keys())

    resolution = PATCH_SIZE if USE_PATCH_224 else STATIC_CFG["RESOLUTION"]

    kwargs = dict(
        dataset_dir=str(dataset_dir_effective),
        output_dir=str(out_dir),
        class_names=[target_name],

        resolution=resolution,
        batch_size=STATIC_CFG["BATCH"],
        grad_accum_steps=8,
        epochs=STATIC_CFG["EPOCHS"],
        lr=STATIC_CFG["LR"],
        weight_decay=5e-4,
        dropout=0.1,
        num_queries=STATIC_CFG["NUM_QUERIES"],

        multi_scale=False,
        gradient_checkpointing=True,
        amp=True,
        num_workers=min(NUM_WORKERS, os.cpu_count() or NUM_WORKERS),
        pin_memory=True,
        persistent_workers=True,

        seed=SEED,
        early_stopping=True,
        checkpoint_interval=10,
        run_test=True,
    )

    def maybe(name, value):
        if name in can and value is not None:
            kwargs[name] = value

    maybe("encoder_name", backbone_ckpt)
    maybe("pretrained_backbone", backbone_ckpt)

    if USE_PATCH_224:
        maybe("square_resize_div_64", False)
        maybe("do_random_resize_via_padding", False)
        maybe("random_resize_via_padding", False)
    else:
        maybe("square_resize_div_64", True)
        maybe("do_random_resize_via_padding", False)
        maybe("random_resize_via_padding", False)

    meta = out_dir / "run_meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "static_cfg.json").write_text(json.dumps(STATIC_CFG, indent=2), encoding="utf-8")
    (meta / "probe_setup.json").write_text(json.dumps({
        "target_name": target_name,
        "ssl_ckpt": backbone_ckpt,
        "dataset_dir_effective": str(dataset_dir_effective),
        "train_fraction": TRAIN_FRACTION,
        "fraction_seed": FRACTION_SEED,
        "input_mode": INPUT_MODE,
        "use_patch_224": USE_PATCH_224,
        "patch_size": PATCH_SIZE if USE_PATCH_224 else None,
        "full_resolution": None if USE_PATCH_224 else FULL_RESOLUTION,
        "seed": SEED,
        "session_root": str(SESSION_ROOT),
    }, indent=2), encoding="utf-8")

    print(f"[TRAIN] {model.__class__.__name__} — {target_name} — SSL={Path(backbone_ckpt).name} → {out_dir}")
    model.train(**kwargs)

    best = find_best_val(out_dir)
    (out_dir / "val_best_summary.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


def _write_class_outputs(out_root: Path, leaderboard: list[dict]) -> dict | None:
    def sort_key(r):
        a = r["val_AP50"]; b = r["val_mAP5095"]
        return (-(a if a is not None else -1), -(b if b is not None else -1))

    leaderboard.sort(key=sort_key)

    csv_path = out_root / "ssl_probe_leaderboard.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(leaderboard[0].keys()) if leaderboard else [])
        w.writeheader()
        for r in leaderboard:
            w.writerow(r)

    best_row = leaderboard[0] if leaderboard else None
    (out_root / "ssl_probe_best.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")
    return best_row


def _run_worker_from_env() -> int:
    target_name = os.environ["RFDETR_WORKER_TARGET_NAME"]
    dataset_dir_effective = Path(os.environ["RFDETR_WORKER_DATASET_DIR"])
    out_dir = Path(os.environ["RFDETR_WORKER_OUT_DIR"])
    ckpt = Path(os.environ["RFDETR_WORKER_BACKBONE_CKPT"])
    run_idx = int(os.environ.get("RFDETR_WORKER_RUN_IDX", "1"))
    run_total = int(os.environ.get("RFDETR_WORKER_RUN_TOTAL", "1"))

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] {target_name}: checkpoint {run_idx}/{run_total} -> {ckpt.name}")

    t0 = time.time()
    best = train_one_run(
        target_name=target_name,
        dataset_dir_effective=dataset_dir_effective,
        out_dir=out_dir,
        backbone_ckpt=str(ckpt),
    )
    dur = round(time.time() - t0, 2)

    row = _build_probe_row(
        idx=run_idx,
        total=run_total,
        target_name=target_name,
        ckpt=ckpt,
        run_dir=out_dir,
        best=best,
        dur=dur,
        dataset_dir_effective=dataset_dir_effective,
    )
    (out_dir / "probe_record.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(
        f"[DONE] {target_name} {run_idx}/{run_total} "
        f"SSL={ckpt.name}  sec={dur:.1f}  AP50={row['val_AP50']}  mAP50-95={row['val_mAP5095']}"
    )
    return 0


def _run_parallel_ckpt_jobs(target_name: str, dataset_dir_effective: Path, jobs: list[dict]) -> list[dict]:
    slots = _gpu_slots_for_parallel()
    print(f"[PARALLEL] {target_name}: GPU slots={slots}  jobs={len(jobs)}")
    if len(slots) <= 1:
        rows: list[dict] = []
        total = len(jobs)
        for j in jobs:
            print(f"\n[RUN] {target_name}: checkpoint {j['idx']}/{total} -> {j['ckpt'].name}")
            t0 = time.time()
            best = train_one_run(
                target_name=target_name,
                dataset_dir_effective=dataset_dir_effective,
                out_dir=j["run_dir"],
                backbone_ckpt=str(j["ckpt"]),
            )
            dur = round(time.time() - t0, 2)
            row = _build_probe_row(
                idx=j["idx"],
                total=total,
                target_name=target_name,
                ckpt=j["ckpt"],
                run_dir=j["run_dir"],
                best=best,
                dur=dur,
                dataset_dir_effective=dataset_dir_effective,
            )
            (j["run_dir"] / "probe_record.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
            print(
                f"[DONE] {target_name} {j['idx']}/{total} "
                f"SSL={j['ckpt'].name}  sec={dur:.1f}  AP50={row['val_AP50']}  mAP50-95={row['val_mAP5095']}"
            )
        return rows

    queue = jobs[:]
    active: dict[str | None, dict] = {}
    failures: list[dict] = []

    while queue or active:
        # launch on free slots
        for slot in slots:
            if slot in active or not queue:
                continue
            j = queue.pop(0)
            env = os.environ.copy()
            env["RFDETR_WORKER_MODE"] = "1"
            env["RFDETR_WORKER_TARGET_NAME"] = target_name
            env["RFDETR_WORKER_DATASET_DIR"] = str(dataset_dir_effective)
            env["RFDETR_WORKER_OUT_DIR"] = str(j["run_dir"])
            env["RFDETR_WORKER_BACKBONE_CKPT"] = str(j["ckpt"])
            env["RFDETR_WORKER_RUN_IDX"] = str(j["idx"])
            env["RFDETR_WORKER_RUN_TOTAL"] = str(len(jobs))
            env["RFDETR_SESSION_ROOT"] = str(SESSION_ROOT)
            env["RFDETR_SESSION_ID"] = str(SESSION_ID)
            if slot is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(slot)

            print(f"[PARALLEL] launch slot={slot} run={j['idx']}/{len(jobs)} ckpt={j['ckpt'].name}")
            p = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve())],
                cwd=str(Path(__file__).resolve().parent),
                env=env,
            )
            active[slot] = {"proc": p, "job": j}

        # poll completions
        finished_slots = []
        for slot, info in active.items():
            rc = info["proc"].poll()
            if rc is None:
                continue
            j = info["job"]
            if rc != 0:
                failures.append({"slot": slot, "job": j, "returncode": rc})
                print(f"[PARALLEL][FAIL] slot={slot} run={j['idx']}/{len(jobs)} ckpt={j['ckpt'].name} rc={rc}")
            else:
                print(f"[PARALLEL][OK] slot={slot} run={j['idx']}/{len(jobs)} ckpt={j['ckpt'].name}")
            finished_slots.append(slot)
        for slot in finished_slots:
            active.pop(slot, None)

        if active:
            time.sleep(2)

    if failures:
        raise RuntimeError(f"{len(failures)} parallel worker(s) failed")

    rows: list[dict] = []
    for j in jobs:
        rec = j["run_dir"] / "probe_record.json"
        if not rec.exists():
            raise FileNotFoundError(f"Missing probe_record.json for run: {j['run_dir']}")
        rows.append(json.loads(rec.read_text(encoding="utf-8")))
    return rows


def run_probe_for_class(session_root: Path, target_name: str, dataset_dir: Path) -> dict:
    class_token = (
        "Epithelial" if "Epithelial" in target_name else
        "Leucocytes" if "Leucocyte" in target_name else
        target_name.replace(" ", "")
    )
    out_root = session_root / class_token
    out_root.mkdir(parents=True, exist_ok=True)

    # Build identical dataset once (cached under OUTPUT_ROOT)
    data_dir_effective = get_or_build_probe_dataset(dataset_dir, OUTPUT_ROOT)

    jobs = []
    for i, ckpt in enumerate(SSL_BACKBONES, start=1):
        run_dir = out_root / f"SSL_{i:02d}__{ckpt.stem}"
        run_dir.mkdir(parents=True, exist_ok=True)
        jobs.append({"idx": i, "ckpt": ckpt, "run_dir": run_dir})

    leaderboard = _run_parallel_ckpt_jobs(target_name, data_dir_effective, jobs)
    best_row = _write_class_outputs(out_root, leaderboard)

    return {"best": best_row, "leaderboard": leaderboard, "out_dir": str(out_root), "dataset_effective": str(data_dir_effective)}

def find_latest_dataset(root: Path, prefix: str) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"STAT_DATASETS_ROOT does not exist: {root}")
    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    )
    if not candidates:
        raise FileNotFoundError(
            f"No dataset starting with '{prefix}' found in {root}"
        )
    return candidates[-1]  # newest lexicographically (timestamped folders)


def resolve_target_datasets() -> dict[str, Path]:
    ds_leu = (
        Path(DATASET_LEUCO_DIR_OVERRIDE)
        if DATASET_LEUCO_DIR_OVERRIDE
        else find_latest_dataset(STAT_DATASETS_ROOT, DATASET_PREFIX_LEU)
    )
    ds_epi = (
        Path(DATASET_EPI_DIR_OVERRIDE)
        if DATASET_EPI_DIR_OVERRIDE
        else find_latest_dataset(STAT_DATASETS_ROOT, DATASET_PREFIX_EPI)
    )
    return {
        "Leucocyte": ds_leu,
        "Squamous Epithelial Cell": ds_epi,
    }

def main():
    if os.getenv("RFDETR_WORKER_MODE", "").strip() == "1":
        _run_worker_from_env()
        return

    # IMPORTANT: do NOT create a new session root here.
    # Use the already-created SESSION_ROOT/SESSION_ID at top of file.
    session_root = SESSION_ROOT
    session_id   = SESSION_ID

    all_datasets = resolve_target_datasets()
    print("[DATASETS]")
    print("  STAT_DATASETS_ROOT:", STAT_DATASETS_ROOT)
    print("  Leucocyte:", all_datasets["Leucocyte"])
    print("  Epithelial:", all_datasets["Squamous Epithelial Cell"])

    selected: list[tuple[str, Path]] = []
    if PROBE_TARGET in ("leu", "all"):
        selected.append(("Leucocyte", all_datasets["Leucocyte"]))
    if PROBE_TARGET in ("epi", "all"):
        selected.append(("Squamous Epithelial Cell", all_datasets["Squamous Epithelial Cell"]))
    if not selected:
        raise RuntimeError("RFDETR_PROBE_TARGET must be one of: leu, epi, all")

    # sanity: dataset splits exist for selected targets
    # NOTE: we only require train+valid, like your HPO script
    #       (test may exist but not required)
    for _, p in selected:
        for part in ("train", "valid"):
            if not (p / part / "_annotations.coco.json").exists():
                raise FileNotFoundError(f"Missing {part} split in {p}")

    print(f"[PLAN] SSL checkpoints: {len(SSL_BACKBONES)}")
    print(f"[PLAN] Targets: {[name for name, _ in selected]}")
    print(f"[PLAN] Total RF-DETR runs: {len(SSL_BACKBONES) * len(selected)}")
    print(f"[PLAN] Input mode: {INPUT_MODE} (patch={USE_PATCH_224}, full_resolution={FULL_RESOLUTION})")
    print(f"[PLAN] Parallel GPU slots: {_gpu_slots_for_parallel()}")
    print("[PLAN] Train subset is built once per target and reused for every SSL checkpoint.")

    results = {}
    for target_name, ds in selected:
        results[target_name] = run_probe_for_class(session_root, target_name, ds)

    final = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "session_root": str(session_root),
        "work_root": str(WORK_ROOT),
        "ssl_ckpt_root": str(SSL_CKPT_ROOT),
        "ssl_ckpts": [str(p) for p in SSL_BACKBONES],
        "stat_datasets_root": str(STAT_DATASETS_ROOT),
        "dataset_prefix_leu": DATASET_PREFIX_LEU,
        "dataset_prefix_epi": DATASET_PREFIX_EPI,
        "dataset_leuco_override": DATASET_LEUCO_DIR_OVERRIDE or None,
        "dataset_epi_override": DATASET_EPI_DIR_OVERRIDE or None,
        "static_cfg": STATIC_CFG,
        "train_fraction": TRAIN_FRACTION,
        "fraction_seed": FRACTION_SEED,
        "seed": SEED,
        "input_mode": INPUT_MODE,
        "use_patch_224": USE_PATCH_224,
        "patch_size": PATCH_SIZE if USE_PATCH_224 else None,
        "full_resolution": None if USE_PATCH_224 else FULL_RESOLUTION,
        "results": {k: v["best"] for k, v in results.items()},
    }

    summary_path = session_root / "FINAL_SSL_PROBE_SUMMARY.json"
    summary_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("\n[FINAL] Summary →", summary_path)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
