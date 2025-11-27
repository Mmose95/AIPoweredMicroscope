# LightlyTrainSSL.py
import os, glob, os.path as op, shutil
from pathlib import Path

from lightly_train import train as lightly_train

from PIL import Image
import json
import numpy as np
import cv2
import concurrent.futures
import re
from collections import defaultdict
import random  # NEW

PATCH_SIZE = 224

# How many CPU workers to use for crop generation
NUM_PATCH_WORKERS = int(os.getenv("SSL_PATCH_WORKERS", "14"))

# NEW: sampling behaviour knobs (all env-configurable)
PATCH_STRIDE_FACTOR = float(os.getenv("SSL_PATCH_STRIDE_FACTOR", "1.0"))
# 1.0  → non-overlapping (old behaviour)
# 0.5  → 50% overlap, more variety

EXTRA_RANDOM_PATCHES = int(os.getenv("SSL_EXTRA_RANDOM_PER_IMAGE", "4"))
# How many extra random crops per full image (after grid). 0 = disable.

# Filtering knobs
BORDERLINE_KEEP_PROB = float(os.getenv("SSL_BORDERLINE_KEEP_PROB", "0.2"))
MEAN_MARGIN_STRICT   = float(os.getenv("SSL_MEAN_MARGIN_STRICT", "5.0"))
MEAN_MARGIN_LOOSE    = float(os.getenv("SSL_MEAN_MARGIN_LOOSE", "10.0"))

STRICT_STD_FACTOR = float(os.getenv("SSL_STRICT_STD_FACTOR", "0.6"))
STRICT_LAP_FACTOR = float(os.getenv("SSL_STRICT_LAP_FACTOR", "0.6"))

LOOSE_STD_FACTOR  = float(os.getenv("SSL_LOOSE_STD_FACTOR", "0.3"))
LOOSE_LAP_FACTOR  = float(os.getenv("SSL_LOOSE_LAP_FACTOR", "0.3"))

# For resolving image paths from COCO JSONs
VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
WINDOWS_PATH_RE = re.compile(r"^[a-zA-Z]:\\")

# ─────────────────────────────────────────────
# 1) Detect USER_BASE_DIR (AAU / SDU style)
# ─────────────────────────────────────────────
def _detect_user_base():
    aau = glob.glob("/work/Member Files:*")
    if aau:
        return op.basename(aau[0])
    sdu = [d for d in glob.glob("/work/*#*") if op.isdir(d)]
    return op.basename(sdu[0]) if sdu else None


USER_BASE_DIR = os.environ.get("USER_BASE_DIR") or _detect_user_base()
if not USER_BASE_DIR:
    raise RuntimeError("Could not determine USER_BASE_DIR")
os.environ["USER_BASE_DIR"] = USER_BASE_DIR

WORK_ROOT = Path("/work") / USER_BASE_DIR
REPO_ROOT = Path(__file__).resolve().parent  # e.g. /work/projects/myproj


def env_path(name: str, default: Path) -> Path:
    v = os.getenv(name, "").strip()
    return Path(v) if v else default


# ─────────────────────────────────────────────
# 2) Read SSL root + sample range from env
# ─────────────────────────────────────────────
SSL_TRAINING_ROOT = Path(
    os.getenv(
        "SSL_TRAINING_ROOT",
        WORK_ROOT / "CellScanData" / "Zoom10x - Quality Assessment_Cleaned",
    )
)

SSL_FIRST_SAMPLE = int(os.getenv("SSL_FIRST_SAMPLE", "1"))
SSL_LAST_SAMPLE = int(os.getenv("SSL_LAST_SAMPLE", "999"))

if SSL_FIRST_SAMPLE > SSL_LAST_SAMPLE:
    raise ValueError(
        f"SSL_FIRST_SAMPLE ({SSL_FIRST_SAMPLE}) > SSL_LAST_SAMPLE ({SSL_LAST_SAMPLE})"
    )

print("USER_BASE_DIR:", USER_BASE_DIR)
print("SSL_TRAINING_ROOT:", SSL_TRAINING_ROOT)
print(f"SSL sample range: {SSL_FIRST_SAMPLE}–{SSL_LAST_SAMPLE}")

# This is where the real microscopy images live on UCloud
IMAGES_FALLBACK_ROOT = env_path(
    "IMAGES_FALLBACK_ROOT",
    SSL_TRAINING_ROOT,
)

# ─────────────────────────────────────────────
# 2b) Calibration stats (epi + leu) from supervised datasets
# ─────────────────────────────────────────────
SSL_STATS_ROOT = WORK_ROOT / "SSL_Stats"
CALIB_STATS_JSON = SSL_STATS_ROOT / "calib_patch_stats.json"
SSL_STATS_ROOT.mkdir(parents=True, exist_ok=True)

SUPERVISED_STATS = None  # filled by build_calibration_stats_if_needed()

# Root where RFDETR supervised datasets live
STAT_DATASETS_ROOT = env_path(
    "STAT_DATASETS_ROOT",
    REPO_ROOT / "SOLO_Supervised_RFDETR" / "Stat_Dataset",
)

# ---------- helpers for resolving image paths ----------

def _index_image_paths(root: Path):
    """Index all images under root for fast lookup by relative path and basename."""
    root = root.resolve()
    by_rel = {}
    by_name = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            rel = p.resolve().relative_to(root).as_posix()
            by_rel[rel] = p
            by_name[p.name].append(p)
    return by_rel, by_name


def _resolve_image_path(file_name: str,
                        images_root: Path,
                        by_rel: dict,
                        by_name: dict) -> Path:
    """
    Resolve a COCO file_name to an actual image file on the mounted drive.
    Handles cases like:
      - 'Sample 24/Patches for Sample 24/...tif'
      - old Windows-style absolute paths
    """
    # normalize slashes
    rel = file_name.replace("\\", "/")

    # direct under images_root
    direct = images_root / rel
    if direct.exists():
        return direct

    # lookup by relative key
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

    # Windows absolute paths in JSON → keep basename only
    if WINDOWS_PATH_RE.match(file_name):
        if base in by_name:
            cands = by_name[base]
            cands.sort(key=lambda q: len(str(q)))
            return cands[0]

    raise FileNotFoundError(
        f"[CALIB] Could not resolve image '{file_name}' under {images_root}"
    )

# ---------- COCO + calibration crop building ----------

def _load_coco_split(dataset_dir: Path, split: str = "train"):
    coco_path = dataset_dir / split / "_annotations.coco.json"
    if not coco_path.exists():
        raise FileNotFoundError(f"[CALIB] Missing COCO JSON: {coco_path}")
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data.get("images", [])}
    anns_by_img = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        anns_by_img.setdefault(img_id, []).append(ann)
    return images, anns_by_img


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_calib_crops_for_dataset(
    dataset_dir: Path,
    crops_dir: Path,
    images_root: Path,
    by_rel: dict,
    by_name: dict,
) -> list[Path]:
    """
    For each annotation in the train split of dataset_dir,
    make one 224×224 crop centred on the bounding box.
    """
    print(f"[CALIB] Building calibration crops from {dataset_dir}")
    images, anns_by_img = _load_coco_split(dataset_dir, split="train")
    print(f"[CALIB]  Images: {len(images)}, anns: {sum(len(v) for v in anns_by_img.values())}")

    crops_dir.mkdir(parents=True, exist_ok=True)

    crop_paths: list[Path] = []

    for img_id, im in images.items():
        if img_id not in anns_by_img:
            continue

        file_name = im["file_name"]
        try:
            img_path = _resolve_image_path(file_name, images_root, by_rel, by_name)
        except FileNotFoundError as e:
            print("[CALIB][WARN]", e)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("[CALIB][WARN] Failed to open", img_path, "error:", e)
            continue

        W, H = img.size
        anns = anns_by_img[img_id]

        for ann_idx, ann in enumerate(anns):
            # COCO bbox is [x, y, w, h]
            x, y, w, h = ann["bbox"]
            cx = x + w / 2.0
            cy = y + h / 2.0

            half = PATCH_SIZE / 2.0
            x0 = int(_clamp(cx - half, 0, W - PATCH_SIZE))
            y0 = int(_clamp(cy - half, 0, H - PATCH_SIZE))
            box = (x0, y0, x0 + PATCH_SIZE, y0 + PATCH_SIZE)

            patch = img.crop(box)

            out_name = f"ds{dataset_dir.name}_img{img_id}_ann{ann_idx}.png"
            out_path = crops_dir / out_name
            patch.save(out_path)
            crop_paths.append(out_path)

    print(f"[CALIB] Saved {len(crop_paths)} calibration crops to {crops_dir}")
    return crop_paths


def compute_stats_from_patch_paths(crop_paths: list[Path]) -> dict:
    means, stds, laps = [], [], []

    for p in crop_paths:
        img = Image.open(p).convert("RGB")
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        means.append(float(gray.mean()))
        stds.append(float(gray.std()))
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        laps.append(lap_var)

    def percentiles(vals, ps=(1, 5, 10, 50, 90, 95, 99)):
        v = np.array(vals, dtype=np.float32)
        return {f"p{p}": float(np.percentile(v, p)) for p in ps}

    stats = {
        "mean": percentiles(means),
        "std": percentiles(stds),
        "lap_var": percentiles(laps),
        "n_crops": len(crop_paths),
        "patch_size": PATCH_SIZE,
    }
    return stats


def _autofind_dataset(root: Path, token: str) -> Path:
    """
    Find the *latest* directory under `root` whose name contains `token`.
    E.g. token='Leucocyte_OVR' will match:
      QA-2025v1_Leucocyte_OVR_20251122-145858
    """
    if not root.is_dir():
        raise FileNotFoundError(f"[CALIB] STAT_DATASETS_ROOT does not exist: {root}")
    cands = sorted(p for p in root.glob(f"*{token}*") if p.is_dir())
    if not cands:
        raise FileNotFoundError(f"[CALIB] No dataset dirs matching '{token}' under {root}")
    return cands[-1]  # newest by name


def build_calibration_stats_if_needed():
    """
    If calib_patch_stats.json is missing, build it from BOTH
    epithelial and leucocyte supervised datasets (if present),
    using crops centred on annotated objects.
    """
    global SUPERVISED_STATS

    if CALIB_STATS_JSON.exists():
        SUPERVISED_STATS = json.loads(CALIB_STATS_JSON.read_text(encoding="utf-8"))
        print("[SSL] Loaded supervised stats from:", CALIB_STATS_JSON)
        print("[SSL] n_crops:", SUPERVISED_STATS.get("n_crops"))
        return

    print("[SSL CALIB] calib_patch_stats.json not found → building from supervised datasets")

    # allow explicit override via env vars
    epi_env = os.getenv("SUPERVISED_DATASET_DIR_EPI", "").strip()
    leu_env = os.getenv("SUPERVISED_DATASET_DIR_LEU", "").strip()

    if epi_env:
        epi_dir = Path(epi_env)
    else:
        try:
            epi_dir = _autofind_dataset(STAT_DATASETS_ROOT, "SquamousEpithelialCell_OVR")
        except FileNotFoundError as e:
            print("[SSL CALIB][WARN]", e)
            epi_dir = None

    if leu_env:
        leu_dir = Path(leu_env)
    else:
        try:
            leu_dir = _autofind_dataset(STAT_DATASETS_ROOT, "Leucocyte_OVR")
        except FileNotFoundError as e:
            print("[SSL CALIB][WARN]", e)
            leu_dir = None

    dataset_dirs = [d for d in (epi_dir, leu_dir) if d is not None and d.is_dir()]

    if not dataset_dirs:
        print("[SSL CALIB][WARN] No supervised datasets found (epi/leu). "
              "Skipping calibration → no filtering will be applied.")
        SUPERVISED_STATS = None
        return

    # Build a global index over the real microscopy images
    if not IMAGES_FALLBACK_ROOT.exists():
        print("[SSL CALIB][WARN] IMAGES_FALLBACK_ROOT does not exist:", IMAGES_FALLBACK_ROOT)
        SUPERVISED_STATS = None
        return

    by_rel, by_name = _index_image_paths(IMAGES_FALLBACK_ROOT)

    all_crops: list[Path] = []

    for ds in dataset_dirs:
        tag = ds.name
        crops_dir = SSL_STATS_ROOT / f"CalibCrops_224_{tag}"
        existing = list(crops_dir.glob("*.png"))
        if existing:
            print(f"[CALIB] Reusing {len(existing)} crops in {crops_dir}")
            crop_paths = existing
        else:
            crop_paths = make_calib_crops_for_dataset(
                ds,
                crops_dir,
                IMAGES_FALLBACK_ROOT,
                by_rel,
                by_name,
            )
        all_crops.extend(crop_paths)

    if not all_crops:
        print("[SSL CALIB][WARN] No calibration crops produced; skipping filtering.")
        SUPERVISED_STATS = None
        return

    stats = compute_stats_from_patch_paths(all_crops)
    CALIB_STATS_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    SUPERVISED_STATS = stats
    print("[SSL CALIB] Saved stats JSON →", CALIB_STATS_JSON)
    print(json.dumps(stats, indent=2))


# build / load stats on import
build_calibration_stats_if_needed()

# ─────────────────────────────────────────────
# 3) Helpers to generate 224×224 crops per sample
# ─────────────────────────────────────────────

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


def compute_patch_stats_from_pil(pil_img):
    """Compute mean, std, and Laplacian variance on a 224×224 PIL patch."""
    arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return mean, std, lap


def should_keep_patch(pil_img, stats: dict | None, rng: random.Random | None = None) -> bool:
    """
    Decide whether to keep a patch based on supervised 224×224 stats.
    If stats is None → keep everything.

    NEW:
    - "Strict" region: keep everything that looks like typical annotated crops.
    - "Borderline" region: keep with probability BORDERLINE_KEEP_PROB to add variation
      (background, mucus, weird texture) without flooding with noise.
    """
    if stats is None:
        return True

    if rng is None:
        rng = random

    mean, std, lap = compute_patch_stats_from_pil(pil_img)

    m_p1  = stats["mean"]["p1"]
    m_p99 = stats["mean"]["p99"]
    s_p10 = stats["std"]["p10"]
    l_p10 = stats["lap_var"]["p10"]

    # 1) Strict: well-behaved patches near annotated stats → always keep
    if (m_p1 - MEAN_MARGIN_STRICT <= mean <= m_p99 + MEAN_MARGIN_STRICT and
        std >= STRICT_STD_FACTOR * s_p10 and
        lap >= STRICT_LAP_FACTOR * l_p10):
        return True

    # 2) Borderline band: a bit darker/brighter / lower texture,
    # but still not completely empty → keep with some probability
    in_loose_mean = (m_p1 - MEAN_MARGIN_LOOSE <= mean <= m_p99 + MEAN_MARGIN_LOOSE)
    in_loose_std  = std >= LOOSE_STD_FACTOR * s_p10
    in_loose_lap  = lap >= LOOSE_LAP_FACTOR * l_p10

    if in_loose_mean and in_loose_std and in_loose_lap:
        if rng.random() < BORDERLINE_KEEP_PROB:
            return True

    # 3) Everything else: discard (likely too bright/dark/flat/blurry)
    return False


def _process_image_for_crops(
    img_path: Path,
    crops_dir: Path,
    patch_size: int,
    stats: dict | None,
) -> tuple[int, int]:
    """
    Worker function:
    - Opens one full-size image
    - Generates grid 224×224 crops (with configurable stride)
    - Adds a few random 224×224 crops per image
    - Filters them using supervised stats
    - Saves accepted crops into crops_dir

    Returns (kept_count, candidate_count).
    """
    from PIL import Image as _Image  # safe to import inside worker too

    img = _Image.open(img_path).convert("RGB")
    w, h = img.size

    # NEW: stride factor for overlap
    stride = max(1, int(round(patch_size * PATCH_STRIDE_FACTOR)))
    xs = _compute_positions(w, patch_size, stride)
    ys = _compute_positions(h, patch_size, stride)

    stem = img_path.stem

    kept = 0
    cand = 0

    # NEW: deterministic RNG per image to make selection reproducible
    rng = random.Random(stem)

    # --- grid-based patches ---
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            box = (x, y, x + patch_size, y + patch_size)
            patch = img.crop(box)
            cand += 1

            if not should_keep_patch(patch, stats, rng=rng):
                continue

            out_name = f"{stem}_y{yi:02d}_x{xi:02d}.png"
            out_path = crops_dir / out_name
            patch.save(out_path)
            kept += 1

    # --- extra random patches for more variation ---
    if EXTRA_RANDOM_PATCHES > 0 and w >= patch_size and h >= patch_size:
        for j in range(EXTRA_RANDOM_PATCHES):
            x = rng.randint(0, w - patch_size)
            y = rng.randint(0, h - patch_size)
            box = (x, y, x + patch_size, y + patch_size)
            patch = img.crop(box)
            cand += 1

            if not should_keep_patch(patch, stats, rng=rng):
                continue

            out_name = f"{stem}_rand{j:02d}.png"
            out_path = crops_dir / out_name
            patch.save(out_path)
            kept += 1

    return kept, cand


def ensure_ssl_crops_for_sample(sample_dir: Path, patch_size: int = PATCH_SIZE) -> Path:
    """
    Ensure we have a folder with 224×224 crops for this Sample.
    If it doesn't exist (or is empty), create it from the full-size images
    using parallel workers.

    Returns the path to the crops folder.
    """
    sample_name = sample_dir.name  # e.g. "Sample 105"
    crops_dir = sample_dir / f"SSLCrops ({patch_size}x{patch_size}) for {sample_name}"

    # If crops already exist and contain images, reuse them
    if crops_dir.exists():
        num_existing = sum(
            1
            for p in crops_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if num_existing > 0:
            print(f"[ensure_ssl_crops_for_sample] Reusing {num_existing} crops in {crops_dir}")
            return crops_dir
    else:
        crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ensure_ssl_crops_for_sample] Creating crops in {crops_dir}")

    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    # Only use full-size images directly inside the Sample folder
    image_files = [
        p for p in sample_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]

    if not image_files:
        raise RuntimeError(f"No full-size images found in {sample_dir}")

    # Parallel over images
    total_kept = 0
    total_cand = 0

    if NUM_PATCH_WORKERS <= 1 or len(image_files) == 1:
        # fallback: single-process
        for img_path in image_files:
            kept, cand = _process_image_for_crops(
                img_path,
                crops_dir,
                patch_size,
                SUPERVISED_STATS,
            )
            total_kept += kept
            total_cand += cand
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PATCH_WORKERS) as ex:
            futures = [
                ex.submit(
                    _process_image_for_crops,
                    img_path,
                    crops_dir,
                    patch_size,
                    SUPERVISED_STATS,
                )
                for img_path in image_files
            ]
            for fut in concurrent.futures.as_completed(futures):
                kept, cand = fut.result()
                total_kept += kept
                total_cand += cand

    print(
        f"[ensure_ssl_crops_for_sample] Saved {total_kept} / {total_cand} crops "
        f"in {crops_dir} (workers={NUM_PATCH_WORKERS})"
    )
    return crops_dir


# ─────────────────────────────────────────────
# 4) Build aggregated dataset from all crop folders
# ─────────────────────────────────────────────
def build_ssl_selection_dataset(
    ssl_root: Path,
    first_sample: int,
    last_sample: int,
) -> Path:
    """
    For each 'Sample XX' in [first_sample, last_sample]:
      - ensure we have 224×224 crops in 'SSLCrops (224x224) for Sample XX'
      - then symlink/copy them into a flat temp dir used as Lightly 'data' root.

    If the temp dir already exists, we simply reuse it to avoid
    expensive deletion/rebuild on UCloud.
    """
    tmp_root = ssl_root.parent / "SSL_LightlySelection_224crops"

    if tmp_root.exists():
        print(f"[build_ssl_selection_dataset] Reusing existing temp dir: {tmp_root}")
        return tmp_root

    print(f"[build_ssl_selection_dataset] Creating new temp dir: {tmp_root}")
    tmp_root.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    total = 0
    missing_samples = []

    for i in range(first_sample, last_sample + 1):
        sample_dir = ssl_root / f"Sample {i}"
        if not sample_dir.is_dir():
            missing_samples.append(i)
            continue

        # Create/use crop folder for this sample
        crops_dir = ensure_ssl_crops_for_sample(sample_dir, patch_size=PATCH_SIZE)

        for p in crops_dir.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue

            dst = tmp_root / f"{i:03d}_{p.name}"
            try:
                dst.symlink_to(p)
            except OSError:
                shutil.copy2(p, dst)
            total += 1

    if missing_samples:
        print("[build_ssl_selection_dataset] Missing Sample folders (ignored):", missing_samples)

    if total == 0:
        raise RuntimeError(
            f"No 224x224 crops found/created in {ssl_root} for samples {first_sample}–{last_sample}"
        )

    print(f"[build_ssl_selection_dataset] Collected {total} crops into {tmp_root}")
    return tmp_root



from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# ─────────────────────────────────────────────
# 5) Main: run Lightly DINOv2 SSL on the crops
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Build patch-only selection dataset
    data_dir = build_ssl_selection_dataset(
        SSL_TRAINING_ROOT,
        SSL_FIRST_SAMPLE,
        SSL_LAST_SAMPLE,
    )

    lightly_train(
        overwrite =True,
        out="outputLightly",
        data=str(data_dir),
        model="dinov2/vitb14",
        method="dinov2",

        accelerator="gpu",
        devices="auto",
        strategy="ddp_find_unused_parameters_false",

        precision="bf16-mixed",
        batch_size=32,
        num_workers=8,
        loader_args=dict(
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
        ),

        # ⭐️ ONLY save every 5 epochs, nothing else ⭐️
        callbacks=dict(
            model_checkpoint=dict(
                filename="epoch_{epoch:03d}",
                every_n_epochs=5,
                save_top_k=-1,
                save_weights_only=True,
            ),
        ),

        trainer_args=dict(
            max_epochs=100,
        ),

        # Use default DINOv2 transforms
        transform_args=None,
    )
