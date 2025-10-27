import sys
import os
import csv
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image

# ----------------- USER SETTINGS -----------------
ROOT_DIR = Path(r"D:\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment_Cleaned")

# Process these samples (inclusive)
SAMPLE_MIN = 67
SAMPLE_MAX = 67

# If a subfolder starting with this prefix exists (e.g., "2025-08-14 10-42-36"),
# use it as the source; otherwise use files directly in "Sample XX".
SUBDIR_PREFIX = "2025"

SAVE_FORMAT = "tif"            # "tif" or "png"
TIFF_COMPRESSION = "raw"       # "raw", "lzw", or "deflate" (only if SAVE_FORMAT="tif")
SKIP_EXISTING = False          # True: don't overwrite existing patches
WRITE_MANIFEST = True          # Write patch_manifest.csv at ROOT_DIR

# Parallelism
MAX_WORKERS: Optional[int] = max(1, (os.cpu_count() or 4) - 1)  # you can set a fixed int if you want
# -------------------------------------------------

# Tiling layout: 3x2 grid (1920x1200 -> six 640x640 with 80 px vertical overlap)
X_STARTS = [0, 640, 1280]
Y_STARTS = [0, 560]  # 80 px overlap
TILE_W = 640
TILE_H = 640

TIFF_EXTS = {".tif", ".tiff"}
SAMPLE_RE = re.compile(r"^Sample\s+(\d+)$", re.IGNORECASE)


def iter_sample_folders(root: Path):
    """Yield (sample_num, path) for Sample folders, filtered to SAMPLE_MIN..SAMPLE_MAX."""
    numbered: List[Tuple[int, Path]] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = SAMPLE_RE.match(p.name.strip())
        if m:
            num = int(m.group(1))
            if SAMPLE_MIN <= num <= SAMPLE_MAX:
                numbered.append((num, p))
    numbered.sort(key=lambda t: t[0])
    for num, p in numbered:
        yield num, p


def get_source_dir(sample_dir: Path) -> Tuple[Path, str]:
    """
    Return (dir_to_read_from, timestamp_folder_name_for_manifest).
    If a subfolder starting with SUBDIR_PREFIX exists, use that; else use sample_dir itself.
    """
    ts = next((d for d in sample_dir.iterdir() if d.is_dir() and d.name.startswith(SUBDIR_PREFIX)), None)
    if ts:
        return ts, ts.name
    return sample_dir, sample_dir.name


def collect_tiffs_shallow(folder: Path) -> List[Path]:
    """Collect .tif/.tiff files directly inside 'folder' (non-recursive)."""
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in TIFF_EXTS])


def ensure_fresh_patches_dir(sample_dir: Path) -> Path:
    """
    Delete any existing patches directory in this sample that starts with 'Patches for' (case-insensitive),
    then create a new one named 'Patches for <Sample X>'.
    """
    for d in sample_dir.iterdir():
        if d.is_dir() and d.name.lower().startswith("patches for"):
            shutil.rmtree(d, ignore_errors=True)
    patches_dir = sample_dir / f"Patches for {sample_dir.name}"
    patches_dir.mkdir(parents=True, exist_ok=True)
    return patches_dir


def save_patch(img: Image.Image, out_path: Path) -> None:
    if SAVE_FORMAT == "png":
        img.save(out_path, format="PNG", compress_level=0)
    else:
        if TIFF_COMPRESSION == "raw":
            img.save(out_path, format="TIFF", compression="raw")
        elif TIFF_COMPRESSION == "lzw":
            img.save(out_path, format="TIFF", compression="tiff_lzw")
        elif TIFF_COMPRESSION == "deflate":
            img.save(out_path, format="TIFF", compression="tiff_deflate")
        else:
            raise ValueError(f"Unknown TIFF compression: {TIFF_COMPRESSION}")


def tile_image(im: Image.Image, img_path: Path) -> List[Tuple[Tuple[int, int], Image.Image]]:
    W, H = im.size
    if W < TILE_W or H < TILE_H:
        return []
    patches: List[Tuple[Tuple[int, int], Image.Image]] = []
    for y in Y_STARTS:
        for x in X_STARTS:
            x2, y2 = x + TILE_W, y + TILE_H
            if x2 <= W and y2 <= H:
                crop = im.crop((x, y, x2, y2))
                patches.append(((x, y), crop))
    return patches


def process_one_image(args) -> Tuple[int, int, List[Dict], List[str]]:
    """
    Worker function (runs in a process). Returns:
    (images_count, patches_count, manifest_rows, warnings)
    """
    tif_path_str, patches_dir_str, ts_name = args
    tif_path = Path(tif_path_str)
    patches_dir = Path(patches_dir_str)

    images_count = 0
    patches_count = 0
    manifest_rows: List[Dict] = []
    warnings: List[str] = []

    try:
        with Image.open(tif_path) as im:
            im.load()
            patches = tile_image(im, tif_path)
    except Exception as e:
        warnings.append(f"[ERROR] Failed to read {tif_path}: {e}")
        return (0, 0, [], warnings)

    if not patches:
        # too small / OOB; not an error
        return (0, 0, [], warnings)

    images_count = 1
    stem = tif_path.stem
    suffix = ".png" if SAVE_FORMAT == "png" else ".tif"

    for (x, y), crop in patches:
        out_name = f"{stem}_patch_x{x}_y{y}{suffix}"
        out_path = patches_dir / out_name
        if SKIP_EXISTING and out_path.exists():
            continue
        try:
            save_patch(crop, out_path)
        except Exception as e:
            warnings.append(f"[ERROR] Failed to save {out_path}: {e}")
            continue
        patches_count += 1
        manifest_rows.append({
            "sample": patches_dir.parent.name,           # "Sample X"
            "timestamp_folder": ts_name,
            "source_image": str(tif_path),
            "patch_path": str(out_path),
            "x": x, "y": y,
            "width": TILE_W, "height": TILE_H
        })

    return (images_count, patches_count, manifest_rows, warnings)


def run(root_dir: Path) -> None:
    if not root_dir.exists():
        print(f"[ERROR] Root dir does not exist: {root_dir}", file=sys.stderr)
        return

    total_images = 0
    total_patches = 0
    manifest_rows_all: List[Dict] = []
    warnings_all: List[str] = []

    # Build work list per image (so workers can parallelize at image level)
    work_items: List[Tuple[str, str, str]] = []
    # Also ensure fresh patches folder per sample
    patches_dir_map: Dict[Path, Path] = {}

    for sample_num, sample_dir in iter_sample_folders(root_dir):
        source_dir, ts_name = get_source_dir(sample_dir)
        patches_dir = ensure_fresh_patches_dir(sample_dir)
        patches_dir_map[sample_dir] = patches_dir

        tiff_files = collect_tiffs_shallow(source_dir)
        if not tiff_files:
            warnings_all.append(f"[WARN] No TIFFs found in {source_dir}")
            continue

        for tif_path in tiff_files:
            work_items.append((str(tif_path), str(patches_dir), ts_name))

    if not work_items:
        print("[INFO] Nothing to do.")
        return

    print(f"[INFO] Starting multiprocessing with {MAX_WORKERS} workers on {len(work_items)} images...")

    # Run workers
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one_image, wi) for wi in work_items]
        for fut in as_completed(futures):
            try:
                imgs, patches, rows, warns = fut.result()
            except Exception as e:
                warnings_all.append(f"[ERROR] Worker crashed: {e}")
                continue
            total_images += imgs
            total_patches += patches
            manifest_rows_all.extend(rows)
            warnings_all.extend(warns)

    # Write manifest
    if WRITE_MANIFEST:
        manifest_path = root_dir / "patch_manifest.csv"
        try:
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "sample", "timestamp_folder", "source_image", "patch_path",
                    "x", "y", "width", "height"
                ])
                writer.writeheader()
                writer.writerows(manifest_rows_all)
        except Exception as e:
            print(f"[ERROR] Failed to write manifest: {e}", file=sys.stderr)
        else:
            print(f"[DONE] Manifest: {manifest_path}")

    # Print warnings (if any)
    for w in warnings_all[:50]:
        print(w, file=sys.stderr)
    if len(warnings_all) > 50:
        print(f"...and {len(warnings_all)-50} more warnings.", file=sys.stderr)

    print(f"[DONE] Images processed: {total_images}, patches saved: {total_patches}")


if __name__ == "__main__":
    run(ROOT_DIR)
