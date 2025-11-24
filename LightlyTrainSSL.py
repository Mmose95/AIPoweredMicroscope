# LightlyTrainSSL.py
import os, glob, os.path as op, shutil
from pathlib import Path

from PIL import Image
from lightly_train import train as lightly_train

PATCH_SIZE = 224


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


def ensure_ssl_crops_for_sample(sample_dir: Path, patch_size: int = PATCH_SIZE) -> Path:
    """
    Ensure we have a folder with 224×224 crops for this Sample.
    If it doesn't exist (or is empty), create it from the full-size images.
    Returns the path to the crops folder.
    """
    sample_name = sample_dir.name               # e.g. "Sample 105"
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

    crop_count = 0

    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        xs = _compute_positions(w, patch_size, patch_size)
        ys = _compute_positions(h, patch_size, patch_size)

        stem = img_path.stem

        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                box = (x, y, x + patch_size, y + patch_size)
                patch = img.crop(box)
                out_name = f"{stem}_y{yi:02d}_x{xi:02d}.png"
                out_path = crops_dir / out_name
                patch.save(out_path)
                crop_count += 1

    print(f"[ensure_ssl_crops_for_sample] Saved {crop_count} crops in {crops_dir}")
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
