# LightlyTrainSSL.py
import os, glob, os.path as op, shutil
from pathlib import Path

from PIL import Image
from lightly_train import train as lightly_train

import json
import numpy as np
import cv2
import concurrent.futures

PATCH_SIZE = 224

# How many CPU workers to use for crop generation (can override via env)
NUM_PATCH_WORKERS = int(os.getenv("SSL_PATCH_WORKERS", "14"))

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


def env_path(name: str, default: Path) -> Path:
    v = os.getenv(name, "").strip()
    return Path(v) if v else default

REPO_ROOT = Path(__file__).resolve().parent


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


def _resolve_coco_image_path(file_name: str, dataset_dir: Path) -> Path:
    """
    Handle both:
      - relative paths (e.g. 'Sample 37/patches/...png')
      - absolute paths (if you've run RESOLVED datasets)
    """
    p = Path(file_name)
    if p.is_absolute():
        return p
    train_root = dataset_dir / "train"
    cand = train_root / file_name
    if cand.exists():
        return cand
    if p.exists():
        return p
    raise FileNotFoundError(f"[CALIB] Could not resolve image path for: {file_name}")


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_calib_crops_for_dataset(dataset_dir: Path, crops_dir: Path) -> list[Path]:
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
            img_path = _resolve_coco_image_path(file_name, dataset_dir)
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
    return cands[-1]  # last one = newest by name


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

    # 1) allow explicit override via env vars
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

    all_crops: list[Path] = []

    for ds in dataset_dirs:
        tag = ds.name
        crops_dir = SSL_STATS_ROOT / f"CalibCrops_224_{tag}"
        existing = list(crops_dir.glob("*.png"))
        if existing:
            print(f"[CALIB] Reusing {len(existing)} crops in {crops_dir}")
            crop_paths = existing
        else:
            crop_paths = make_calib_crops_for_dataset(ds, crops_dir)
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


def should_keep_patch(pil_img, stats: dict | None) -> bool:
    """
    Decide whether to keep a patch based on supervised 224×224 stats.
    If stats is None → keep everything.
    """
    if stats is None:
        return True

    mean, std, lap = compute_patch_stats_from_pil(pil_img)

    m_p1 = stats["mean"]["p1"]
    m_p99 = stats["mean"]["p99"]
    s_p10 = stats["std"]["p10"]
    l_p10 = stats["lap_var"]["p10"]

    # Slack margins so we don't over-filter
    mean_margin = 5.0

    # Filter extreme dark/bright patches
    if not (m_p1 - mean_margin <= mean <= m_p99 + mean_margin):
        return False

    # Require at least some texture & focus, but not too strict
    if std < 0.6 * s_p10:
        return False
    if lap < 0.6 * l_p10:
        return False

    return True


def _process_image_for_crops(
    img_path: Path,
    crops_dir: Path,
    patch_size: int,
    stats: dict | None,
) -> tuple[int, int]:
    """
    Worker function:
    - Opens one full-size image
    - Generates all candidate 224×224 crops
    - Filters them using supervised stats
    - Saves accepted crops into crops_dir

    Returns (kept_count, candidate_count).
    """
    from PIL import Image as _Image  # safe to import inside worker too

    img = _Image.open(img_path).convert("RGB")
    w, h = img.size

    xs = _compute_positions(w, patch_size, patch_size)
    ys = _compute_positions(h, patch_size, patch_size)

    stem = img_path.stem

    kept = 0
    cand = 0

    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            box = (x, y, x + patch_size, y + patch_size)
            patch = img.crop(box)
            cand += 1

            if not should_keep_patch(patch, stats):
                continue

            out_name = f"{stem}_y{yi:02d}_x{xi:02d}.png"
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
      - then symlink/copy them into a flat temp dir used as Lightly 'data' root
    """
    tmp_root = ssl_root.parent / "SSL_LightlySelection_224crops"
    if tmp_root.exists():
        print(f"[build_ssl_selection_dataset] Removing old temp dir: {tmp_root}")
        shutil.rmtree(tmp_root)
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


# ─────────────────────────────────────────────
# 5) Main: run Lightly DINOv2 SSL on the crops
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Build patch-only selection for the chosen sample range
    data_dir = build_ssl_selection_dataset(
        SSL_TRAINING_ROOT,
        SSL_FIRST_SAMPLE,
        SSL_LAST_SAMPLE,
    )

    lightly_train(
        out="outputLightly",
        data=str(data_dir),
        model="dinov2_vit/vitb14",   # you can change to dinov2_vit/vits14 if needed
        method="dinov2",

        accelerator="gpu",
        devices="auto",              # let Lightning pick available GPUs
        strategy="ddp_find_unused_parameters_false",

        precision="bf16-mixed",
        batch_size=32,               # per GPU
        num_workers=6,
        loader_args=dict(
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
        ),
        transform_args=dict(
            image_size=[PATCH_SIZE, PATCH_SIZE],  # 224×224 crops
            local_view=dict(
                num_views=2,
                view_size=[98, 98],
                random_resize=dict(min_scale=0.05, max_scale=0.32),
                gaussian_blur=dict(
                    prob=0.5,
                    blur_limit=0,
                    sigmas=[0.1, 2.0],
                ),
            ),
        ),
    )
