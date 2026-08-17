"""Batch-tile whole-slide images for local storage and later CVAT selection.

The default 2584 x 1936 level-0 tile geometry is the calibrated CellScan field
size used elsewhere in this project.  Tiles are laid out from the actual
scanned-area origin recorded in each slide's ``Slidedat.ini``.  No tile is
discarded by default: background statistics are written to the manifest so a
CVAT subset can be selected later without repeating WSI decoding.

Examples
--------
Preview the planned work without writing tiles::

    python tileWSIImages.py --dry-run

Tile every MRXS slide in the default input folder::

    python tileWSIImages.py --output "E:\\PHD\\WSI_tiles"

Resume an interrupted run (already existing tile files are left untouched)::

    python tileWSIImages.py --existing-sample resume
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

# Allow this script to be launched directly from its preprocessing subfolder.
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from viewDICOMTiles import (
    get_ini_float,
    make_geometry,
    rgba_to_rgb_on_white,
    read_slidedat_ini,
    resolve_slide_path,
    select_tile,
)

try:
    import openslide
    from openslide import OpenSlideError
except ImportError as exc:  # pragma: no cover - depends on local installation
    raise SystemExit(
        "OpenSlide is required. Install with: "
        "pip install openslide-python openslide-bin"
    ) from exc


DEFAULT_INPUT = Path(r"E:/Patologi afd. - Aalborg/Original 40x Hammamatsu scans")
DEFAULT_OUTPUT = Path("E:/Patologi afd. - Aalborg/40x Input tiles for CVAT")
CALIBRATED_TILE_WIDTH = 2584
CALIBRATED_TILE_HEIGHT = 1936
DEFAULT_WORKERS = 8
DEFAULT_EXISTING_SAMPLE_ACTION = "skip"  # "skip", "resume", or "overwrite"
SUPPORTED_WSI_EXTENSIONS = (".mrxs", ".ndpi", ".svs", ".tif", ".tiff")
MANIFEST_FIELDS = (
    "slide_id", "source_slide", "file_name", "level", "row", "col",
    "level0_x", "level0_y", "tile_width", "tile_height",
    "valid_width", "valid_height", "downsample", "mpp_x", "mpp_y",
    "mean_intensity", "white_fraction", "status",
)


@dataclass(frozen=True)
class TileRecord:
    slide_id: str
    source_slide: str
    file_name: str
    level: int
    row: int
    col: int
    level0_x: int
    level0_y: int
    tile_width: int
    tile_height: int
    valid_width: int
    valid_height: int
    downsample: float
    mpp_x: Optional[float]
    mpp_y: Optional[float]
    mean_intensity: Optional[float]
    white_fraction: Optional[float]
    status: str


@dataclass(frozen=True)
class TileJob:
    record: TileRecord
    output_path: str


_WORKER_SLIDE = None
_WORKER_FORMAT = "jpg"
_WORKER_JPEG_QUALITY = 95


def close_worker_slide() -> None:
    global _WORKER_SLIDE
    if _WORKER_SLIDE is not None:
        _WORKER_SLIDE.close()
        _WORKER_SLIDE = None


def initialize_worker(slide_path: str, image_format: str, jpeg_quality: int) -> None:
    """Open one independent OpenSlide handle in each worker process."""
    global _WORKER_SLIDE, _WORKER_FORMAT, _WORKER_JPEG_QUALITY
    _WORKER_SLIDE = openslide.OpenSlide(slide_path)
    _WORKER_FORMAT = image_format
    _WORKER_JPEG_QUALITY = jpeg_quality
    atexit.register(close_worker_slide)


def process_tile_job(job: TileJob) -> TileRecord:
    """Read or resume one tile inside a worker and return its manifest record."""
    output_path = Path(job.output_path)
    record = job.record
    if record.status == "existing":
        with Image.open(output_path) as image:
            mean_intensity, white_fraction = image_statistics(image)
    else:
        if _WORKER_SLIDE is None:  # Defensive guard for direct/test invocation.
            raise RuntimeError("Tile worker has no OpenSlide handle.")
        image = _WORKER_SLIDE.read_region(
            (record.level0_x, record.level0_y),
            record.level,
            (record.tile_width, record.tile_height),
        )
        image = rgba_to_rgb_on_white(image)
        mean_intensity, white_fraction = image_statistics(image)
        save_args = {}
        if _WORKER_FORMAT == "jpg":
            save_args = {"quality": _WORKER_JPEG_QUALITY, "subsampling": 0}
        image.save(output_path, **save_args)
    return replace(
        record,
        mean_intensity=mean_intensity,
        white_fraction=white_fraction,
    )


def discover_slides(input_path: Path) -> list[Path]:
    """Return supported WSI files, resolving MRXS companion folders once."""
    input_path = input_path.expanduser().resolve()
    if input_path.is_file() or (
        input_path.is_dir() and input_path.with_suffix(".mrxs").is_file()
    ):
        return [resolve_slide_path(input_path).slide_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    slides = sorted(
        (
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_WSI_EXTENSIONS
        ),
        key=lambda path: str(path).lower(),
    )
    if not slides:
        supported = ", ".join(SUPPORTED_WSI_EXTENSIONS)
        raise FileNotFoundError(
            f"No supported WSI files ({supported}) found under: {input_path}"
        )
    return slides


def tile_filename(slide_id: str, level: int, row: int, col: int, x: int, y: int, extension: str) -> str:
    return f"{slide_id}_L{level}_R{row:05d}_C{col:05d}_X{x}_Y{y}.{extension}"


def image_statistics(image) -> tuple[float, float]:
    """Return cheap brightness measures useful for later tissue/CVAT selection."""
    thumbnail = image.copy()
    thumbnail.thumbnail((256, 256))
    pixels = np.asarray(thumbnail.convert("RGB"), dtype=np.uint8)
    gray = pixels.mean(axis=2)
    return round(float(gray.mean()), 3), round(float((gray >= 245).mean()), 6)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def iter_positions(max_row: int, max_col: int) -> Iterable[tuple[int, int]]:
    for row in range(max_row + 1):
        for col in range(max_col + 1):
            yield row, col


def completed_sample_summary(
    slide_output: Path,
    source_slide: Path,
    level: int,
    tile_width: int,
    tile_height: int,
    image_format: str,
) -> Optional[dict]:
    """Return the prior summary only when its declared output is still complete."""
    summary_path = slide_output / "slide_summary.json"
    manifest_path = slide_output / "tiles.csv"
    if not summary_path.is_file() or not manifest_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected = int(summary["tile_count"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if int(summary.get("level", -1)) != level:
        return None
    if summary.get("tile_size") != [tile_width, tile_height]:
        return None
    try:
        recorded_source = Path(summary["source_slide"]).resolve()
    except (KeyError, TypeError):
        return None
    if recorded_source != source_slide.resolve():
        return None
    actual = sum(1 for _ in slide_output.glob(f"*.{image_format}"))
    return summary if expected > 0 and actual == expected else None


def process_slide(slide_path: Path, args: argparse.Namespace) -> dict[str, object]:
    paths = resolve_slide_path(slide_path)
    slide_id = paths.slide_path.stem
    slide_output = args.output / slide_id
    manifest_path = slide_output / "tiles.csv"
    started = time.monotonic()

    if args.existing_sample == "skip":
        previous = completed_sample_summary(
            slide_output=slide_output,
            source_slide=paths.slide_path,
            level=args.level,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            image_format=args.format,
        )
        if previous is not None:
            print(
                f"[SKIP] {slide_id}: completed tiling already exists "
                f"({int(previous['tile_count']):,} tiles)"
            )
            return {**previous, "run_status": "skipped_existing_sample"}

    with openslide.OpenSlide(str(paths.slide_path)) as slide:
        if not 0 <= args.level < slide.level_count:
            raise ValueError(
                f"Slide {slide_id} has levels 0..{slide.level_count - 1}; "
                f"requested {args.level}."
            )
        ini = read_slidedat_ini(paths.data_dir)
        geometry = make_geometry(
            slide, ini, args.level, args.origin, args.tile_width, args.tile_height
        )
        downsample = float(slide.level_downsamples[args.level])
        level_area_width = max(1, math.ceil(geometry.area_width / downsample))
        level_area_height = max(1, math.ceil(geometry.area_height / downsample))
        total = (geometry.max_row + 1) * (geometry.max_col + 1)
        mpp_x = slide.properties.get("openslide.mpp-x") or get_ini_float(
            ini, "LAYER_0_LEVEL_0_SECTION", "MICROMETER_PER_PIXEL_X"
        )
        mpp_y = slide.properties.get("openslide.mpp-y") or get_ini_float(
            ini, "LAYER_0_LEVEL_0_SECTION", "MICROMETER_PER_PIXEL_Y"
        )
        mpp_x = float(mpp_x) if mpp_x is not None else None
        mpp_y = float(mpp_y) if mpp_y is not None else None

        print(
            f"[PLAN] {slide_id}: {geometry.max_col + 1} cols x "
            f"{geometry.max_row + 1} rows = {total:,} tiles; "
            f"area={geometry.area_width}x{geometry.area_height} level-0 px; "
            f"workers={args.workers}"
        )
        if args.dry_run:
            return {"slide_id": slide_id, "planned_tiles": total, "status": "dry-run"}

        slide_output.mkdir(parents=True, exist_ok=True)
        jobs = []
        for row, col in iter_positions(geometry.max_row, geometry.max_col):
            selection = select_tile(slide, geometry, args.level, row, col)
            name = tile_filename(
                slide_id, args.level, row, col,
                selection.base_x, selection.base_y, args.format,
            )
            output_path = slide_output / name
            valid_width = min(
                args.tile_width, level_area_width - col * args.tile_width
            )
            valid_height = min(
                args.tile_height, level_area_height - row * args.tile_height
            )
            status = (
                "existing"
                if args.existing_sample == "resume" and output_path.is_file()
                else "written"
            )
            jobs.append(TileJob(
                record=TileRecord(
                    slide_id=slide_id,
                    source_slide=str(paths.slide_path),
                    file_name=name,
                    level=args.level,
                    row=row,
                    col=col,
                    level0_x=selection.base_x,
                    level0_y=selection.base_y,
                    tile_width=args.tile_width,
                    tile_height=args.tile_height,
                    valid_width=max(0, valid_width),
                    valid_height=max(0, valid_height),
                    downsample=downsample,
                    mpp_x=mpp_x,
                    mpp_y=mpp_y,
                    mean_intensity=None,
                    white_fraction=None,
                    status=status,
                ),
                output_path=str(output_path),
            ))

        written = sum(job.record.status == "written" for job in jobs)
        skipped = total - written
        with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
            writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=initialize_worker,
                initargs=(str(paths.slide_path), args.format, args.jpeg_quality),
            ) as executor:
                records = executor.map(process_tile_job, jobs, chunksize=4)
                for index, record in enumerate(records, 1):
                    writer.writerow(asdict(record))
                    if index % args.progress_every == 0 or index == total:
                        print(f"[TILE] {slide_id}: {index:,}/{total:,}")

        summary = {
            "slide_id": slide_id,
            "source_slide": str(paths.slide_path),
            "output_directory": str(slide_output.resolve()),
            "manifest": str(manifest_path.resolve()),
            "level": args.level,
            "level_downsample": downsample,
            "level0_dimensions": list(slide.dimensions),
            "scanned_area_level0": {
                "x": geometry.origin_x, "y": geometry.origin_y,
                "width": geometry.area_width, "height": geometry.area_height,
            },
            "tile_size": [args.tile_width, args.tile_height],
            "grid": {"rows": geometry.max_row + 1, "columns": geometry.max_col + 1},
            "tile_count": total,
            "workers": args.workers,
            "existing_sample_action": args.existing_sample,
            "tiles_written": written,
            "tiles_already_existing": skipped,
            "mpp_level0": [mpp_x, mpp_y],
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(slide_output / "slide_summary.json", summary)
        return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="WSI file, MRXS data folder, or folder containing slides.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Local root directory for tiles and manifests.")
    parser.add_argument("--level", type=int, default=0, help="OpenSlide pyramid level (default: full resolution level 0).")
    parser.add_argument("--tile-width", type=int, default=CALIBRATED_TILE_WIDTH)
    parser.add_argument("--tile-height", type=int, default=CALIBRATED_TILE_HEIGHT)
    parser.add_argument("--origin", choices=("scanned-area", "slide"), default="scanned-area")
    parser.add_argument("--format", choices=("png", "jpg"), default="jpg", help="JPEG is much smaller for brightfield WSI tiles.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel tile processes (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--existing-sample",
        choices=("skip", "resume", "overwrite"),
        default=DEFAULT_EXISTING_SAMPLE_ACTION,
        help=(
            "How to handle an existing sample folder: skip a completed sample, "
            "resume missing tiles, or overwrite every tile "
            f"(default: {DEFAULT_EXISTING_SAMPLE_ACTION})."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Report tile counts without creating output files.")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args(argv)
    if args.tile_width <= 0 or args.tile_height <= 0:
        parser.error("tile dimensions must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    args.output = args.output.expanduser().resolve()
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        slides = discover_slides(args.input)
        print(f"[INFO] Found {len(slides)} slide(s). Output: {args.output}")
        summaries = [process_slide(path, args) for path in slides]
        if not args.dry_run:
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(args.output / "tiling_summary.json", {
                "configuration": {
                    "input": str(args.input.expanduser().resolve()),
                    "level": args.level,
                    "tile_size": [args.tile_width, args.tile_height],
                    "origin": args.origin,
                    "format": args.format,
                    "workers": args.workers,
                    "existing_sample_action": args.existing_sample,
                },
                "slides": summaries,
            })
        print(f"[DONE] Planned/processed {len(summaries)} slide(s).")
        return 0
    except (FileNotFoundError, ValueError, OpenSlideError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
