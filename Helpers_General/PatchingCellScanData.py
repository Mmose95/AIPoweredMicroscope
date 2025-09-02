# Tile 1920x1200 TIFF images into six 640x640 patches and save under each "Sample X" folder.
# Expected structure:
#   <ROOT_DIR>/
#     Sample 1/
#       2025-08-04 14-13-13/
#         *.tif
#     Sample 2/
#       <timestamp>/
#         *.tif
#
# Patches are written to:
#   Sample X/patches for Sample X/<original>_patch_x{X}_y{Y}.tif
#
# Adjust ROOT_DIR below, then Run in PyCharm.

import sys
import csv
import re
from pathlib import Path
from typing import List, Tuple, Dict

from PIL import Image

# ----------------- USER SETTINGS -----------------
ROOT_DIR = Path(r"\\tsclient\E\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment")
SAVE_FORMAT = "tif"            # "tif" or "png"
TIFF_COMPRESSION = "raw"       # "raw", "lzw", or "deflate" (only used if SAVE_FORMAT="tif")
SKIP_EXISTING = False          # True to avoid overwriting existing patches
WRITE_MANIFEST = True          # Write patch_manifest.csv at ROOT_DIR
# -------------------------------------------------

# Tiling layout: 3x2 grid with slight vertical overlap -> 6 patches of 640x640
X_STARTS = [0, 640, 1280]
Y_STARTS = [0, 560]  # 80 px overlap
TILE_W = 640
TILE_H = 640

TIFF_EXTS = {".tif", ".tiff"}
TIMESTAMP_RE = re.compile(r"^(?:19|20)\d\d-\d\d-\d\d[ _]\d\d-\d\d-\d\d$")  # "YYYY-MM-DD HH-MM-SS"


def is_timestamp_folder(name: str) -> bool:
    return TIMESTAMP_RE.match(name) is not None


def iter_sample_folders(root: Path):
    for p in sorted(root.glob("Sample *")):
        if p.is_dir():
            yield p


def iter_timestamp_folders(sample_dir: Path):
    for p in sorted(sample_dir.iterdir()):
        if p.is_dir() and is_timestamp_folder(p.name):
            yield p


def collect_tiffs(folder: Path) -> List[Path]:
    files: List[Path] = []
    for ext in TIFF_EXTS:
        files.extend(sorted(folder.rglob(f"*{ext}")))
    return files


def ensure_patches_dir(sample_dir: Path) -> Path:
    patches_dir = sample_dir / f"patches for {sample_dir.name}"
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
        print(f"[WARN] Skipping (too small): {img_path} ({W}x{H})", file=sys.stderr)
        return []

    patches: List[Tuple[Tuple[int, int], Image.Image]] = []
    for y in Y_STARTS:
        for x in X_STARTS:
            x2, y2 = x + TILE_W, y + TILE_H
            if x2 <= W and y2 <= H:
                crop = im.crop((x, y, x2, y2))
                patches.append(((x, y), crop))
            else:
                print(f"[WARN] Skipping OOB tile ({x},{y}) for {img_path} of size {W}x{H}", file=sys.stderr)
    return patches


def run(root_dir: Path) -> None:
    if not root_dir.exists():
        print(f"[ERROR] Root dir does not exist: {root_dir}", file=sys.stderr)
        return

    manifest_rows: List[Dict] = []
    total_images = 0
    total_patches = 0

    for sample_dir in iter_sample_folders(root_dir):
        patches_dir = ensure_patches_dir(sample_dir)
        print(f"[INFO] Processing {sample_dir} -> {patches_dir}")

        for ts_dir in iter_timestamp_folders(sample_dir):
            tiff_files = collect_tiffs(ts_dir)
            if not tiff_files:
                print(f"[WARN] No TIFFs found in {ts_dir}", file=sys.stderr)
                continue

            for tif_path in tiff_files:
                try:
                    with Image.open(tif_path) as im:
                        im.load()
                        patches = tile_image(im, tif_path)
                except Exception as e:
                    print(f"[ERROR] Failed to read {tif_path}: {e}", file=sys.stderr)
                    continue

                if not patches:
                    continue

                total_images += 1
                stem = tif_path.stem
                for (x, y), crop in patches:
                    suffix = ".png" if SAVE_FORMAT == "png" else ".tif"
                    out_name = f"{stem}_patch_x{x}_y{y}{suffix}"
                    out_path = patches_dir / out_name

                    if SKIP_EXISTING and out_path.exists():
                        continue

                    try:
                        save_patch(crop, out_path)
                    except Exception as e:
                        print(f"[ERROR] Failed to save {out_path}: {e}", file=sys.stderr)
                        continue

                    total_patches += 1
                    manifest_rows.append({
                        "sample": sample_dir.name,
                        "timestamp_folder": ts_dir.name,
                        "source_image": str(tif_path),
                        "patch_path": str(out_path),
                        "x": x, "y": y,
                        "width": TILE_W, "height": TILE_H
                    })

    if WRITE_MANIFEST:
        manifest_path = root_dir / "patch_manifest.csv"
        try:
            with open(manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "sample", "timestamp_folder", "source_image", "patch_path",
                    "x", "y", "width", "height"
                ])
                writer.writeheader()
                for row in manifest_rows:
                    writer.writerow(row)
        except Exception as e:
            print(f"[ERROR] Failed to write manifest: {e}", file=sys.stderr)
        else:
            print(f"[DONE] Manifest: {manifest_path}")

    print(f"[DONE] Images processed: {total_images}, patches saved: {total_patches}")


if __name__ == "__main__":
    run(ROOT_DIR)
