"""Select diverse candidate tiles from every completed WSI for CVAT upload.

By default, each WSI contributes 250 tiles: 200 likely cell-rich, 35 low-cell,
and 15 negative-like.  These are image-content heuristics, not biological
labels.  Original tile filenames and per-WSI folder names are preserved.

The script is safe to start from PyCharm with no arguments.  It scans completed
tile folders, resumes interrupted copies, and skips WSI selections that already
have a completed ``selection_summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError


DEFAULT_TILES_ROOT = Path(r"E:\Patologi afd. - Aalborg\40x Input tiles for CVAT")
OUTPUT_FOLDER_NAME = "Selected input 40x hammamatsu input tiles"
DEFAULT_PER_WSI = 250
DEFAULT_RICH_FRACTION = 0.80
DEFAULT_NEGATIVE_FRACTION = 0.06
DEFAULT_WORKERS = 8
DEFAULT_SEED = 42
SELECTION_FIELDS = (
    "slide_id", "file_name", "source_path", "selected_path", "category",
    "row", "col", "level0_x", "level0_y", "valid_fraction",
    "mean_gray", "mean_saturation", "dark_fraction", "texture",
    "content_score", "content_percentile",
)


@dataclass(frozen=True)
class TileMetrics:
    file_name: str
    mean_gray: float
    mean_saturation: float
    dark_fraction: float
    texture: float


@dataclass(frozen=True)
class SelectedTile:
    slide_id: str
    file_name: str
    source_path: str
    selected_path: str
    category: str
    row: int
    col: int
    level0_x: int
    level0_y: int
    valid_fraction: float
    mean_gray: float
    mean_saturation: float
    dark_fraction: float
    texture: float
    content_score: float
    content_percentile: float


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def discover_completed_wsi_folders(tiles_root: Path, output_root: Path) -> list[Path]:
    folders = []
    for folder in tiles_root.iterdir():
        if not folder.is_dir() or folder.resolve() == output_root.resolve():
            continue
        manifest = folder / "tiles.csv"
        summary = folder / "slide_summary.json"
        if manifest.is_file() and summary.is_file():
            folders.append(folder)
    return sorted(folders, key=lambda path: path.name.lower())


def image_metrics(path_string: str) -> TileMetrics:
    """Decode a reduced JPEG representation and measure colour/texture content."""
    path = Path(path_string)
    try:
        with Image.open(path) as image:
            image.draft("RGB", (256, 256))
            image = image.convert("RGB")
            image.thumbnail((256, 256))
            pixels = np.asarray(image, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"Cannot read tile: {path}") from exc

    maximum = pixels.max(axis=2)
    minimum = pixels.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1.0 / 255.0)
    gray = pixels.mean(axis=2)
    horizontal = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.0
    vertical = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.0
    return TileMetrics(
        file_name=path.name,
        mean_gray=round(float(gray.mean()), 6),
        mean_saturation=round(float(saturation.mean()), 6),
        dark_fraction=round(float((gray < 0.86).mean()), 6),
        texture=round(float((horizontal + vertical) / 2.0), 6),
    )


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1, len(values) - 1)


def spatially_diverse_pick(
    candidates: list[dict], count: int, seed: int, prefer_high: bool
) -> list[dict]:
    """Greedily select content-ranked tiles while initially avoiding neighbours."""
    rng = random.Random(seed)
    ordered = sorted(
        candidates,
        key=lambda item: (
            item["content_score"] if prefer_high else -item["content_score"],
            rng.random(),
        ),
        reverse=True,
    )
    selected: list[dict] = []
    selected_names: set[str] = set()
    for minimum_distance in (2, 1, 0):
        for item in ordered:
            if item["file_name"] in selected_names:
                continue
            if minimum_distance and any(
                max(abs(item["row"] - other["row"]), abs(item["col"] - other["col"]))
                <= minimum_distance
                for other in selected
            ):
                continue
            selected.append(item)
            selected_names.add(item["file_name"])
            if len(selected) == count:
                return selected
    return selected


def allocate_counts(total: int, rich_fraction: float, negative_fraction: float) -> tuple[int, int, int]:
    rich = round(total * rich_fraction)
    negative = round(total * negative_fraction)
    low = total - rich - negative
    return rich, low, negative


def score_tiles(folder: Path, rows: list[dict[str, str]], workers: int) -> list[dict]:
    eligible = []
    paths = []
    for row in rows:
        tile_width = max(1, int(row["tile_width"]))
        tile_height = max(1, int(row["tile_height"]))
        valid_fraction = (
            int(row["valid_width"]) * int(row["valid_height"])
            / (tile_width * tile_height)
        )
        tile_path = folder / row["file_name"]
        if valid_fraction >= 0.90 and tile_path.is_file():
            parsed = dict(row)
            parsed.update({
                "row": int(row["row"]),
                "col": int(row["col"]),
                "level0_x": int(row["level0_x"]),
                "level0_y": int(row["level0_y"]),
                "valid_fraction": valid_fraction,
                "source_path": str(tile_path),
            })
            eligible.append(parsed)
            paths.append(str(tile_path))

    with ProcessPoolExecutor(max_workers=workers) as executor:
        metrics = list(executor.map(image_metrics, paths, chunksize=16))
    by_name = {metric.file_name: metric for metric in metrics}
    saturation = np.array([by_name[item["file_name"]].mean_saturation for item in eligible])
    darkness = np.array([by_name[item["file_name"]].dark_fraction for item in eligible])
    texture = np.array([by_name[item["file_name"]].texture for item in eligible])
    scores = (
        0.35 * percentile_ranks(saturation)
        + 0.30 * percentile_ranks(darkness)
        + 0.35 * percentile_ranks(texture)
    )
    score_percentiles = percentile_ranks(scores)
    for item, score, percentile in zip(eligible, scores, score_percentiles):
        metric = by_name[item["file_name"]]
        item.update(asdict(metric))
        item["content_score"] = round(float(score), 6)
        item["content_percentile"] = round(float(percentile), 6)
    return eligible


def choose_categories(scored: list[dict], total: int, rich_fraction: float, negative_fraction: float, seed: int) -> list[tuple[str, dict]]:
    total = min(total, len(scored))
    if total == 0:
        return []
    rich_count, low_count, negative_count = allocate_counts(
        total, rich_fraction, negative_fraction
    )

    # If every eligible tile will be used, exact count-based slices avoid
    # percentile-boundary rounding gaps (for example, 90 tiles at 80/14/6%).
    if total == len(scored):
        ordered = sorted(scored, key=lambda item: item["content_score"])
        negative = ordered[:negative_count]
        low_start = negative_count
        low = ordered[low_start:low_start + low_count]
        rich = ordered[low_start + low_count:]
        return (
            [("cell_rich", item) for item in rich]
            + [("low_cell", item) for item in low]
            + [("negative_like", item) for item in negative]
        )

    rich_cutoff = 1.0 - rich_fraction
    negative_cutoff = negative_fraction
    rich_pool = [
        item for item in scored
        if item["content_percentile"] >= rich_cutoff
    ]
    low_pool = [
        item for item in scored
        if negative_cutoff <= item["content_percentile"] < rich_cutoff
    ]
    negative_pool = [
        item for item in scored
        if item["content_percentile"] < negative_cutoff
    ]

    rich = spatially_diverse_pick(rich_pool, rich_count, seed + 1, True)
    used = {item["file_name"] for item in rich}
    low = spatially_diverse_pick(
        [item for item in low_pool if item["file_name"] not in used],
        low_count, seed + 2, True,
    )
    used.update(item["file_name"] for item in low)
    negative = spatially_diverse_pick(
        [item for item in negative_pool if item["file_name"] not in used],
        negative_count, seed + 3, False,
    )
    chosen = [("cell_rich", item) for item in rich]
    chosen += [("low_cell", item) for item in low]
    chosen += [("negative_like", item) for item in negative]
    if len(chosen) != total:
        raise RuntimeError(
            f"Category selection produced {len(chosen)} of {total} requested tiles."
        )
    return chosen


def transfer_file(source: Path, destination: Path, mode: str) -> None:
    if destination.is_file():
        return
    if mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def select_for_wsi(folder: Path, output_root: Path, args: argparse.Namespace) -> dict:
    destination = output_root / folder.name
    summary_path = destination / "selection_summary.json"
    if summary_path.is_file() and not args.reselect:
        print(f"[SKIP] {folder.name}: completed selection already exists")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    started = time.monotonic()
    rows = read_manifest(folder / "tiles.csv")
    print(f"[SCORE] {folder.name}: examining {len(rows):,} tiles")
    scored = score_tiles(folder, rows, args.workers)
    target_count = min(args.per_wsi, len(scored))
    if target_count < args.per_wsi:
        print(
            f"[WARN] {folder.name}: only {target_count:,} eligible tiles are "
            f"available; selecting all of them instead of {args.per_wsi:,}"
        )
    chosen = choose_categories(
        scored, target_count, args.rich_fraction, args.negative_fraction,
        args.seed + sum(ord(char) for char in folder.name),
    )
    if not args.dry_run:
        destination.mkdir(parents=True, exist_ok=True)
    selected_rows = []
    for category, item in chosen:
        source = Path(item["source_path"])
        selected_path = destination / source.name
        if not args.dry_run:
            transfer_file(source, selected_path, args.transfer)
        selected_rows.append(SelectedTile(
            slide_id=folder.name,
            file_name=source.name,
            source_path=str(source),
            selected_path=str(selected_path),
            category=category,
            row=item["row"], col=item["col"],
            level0_x=item["level0_x"], level0_y=item["level0_y"],
            valid_fraction=round(item["valid_fraction"], 6),
            mean_gray=item["mean_gray"],
            mean_saturation=item["mean_saturation"],
            dark_fraction=item["dark_fraction"], texture=item["texture"],
            content_score=item["content_score"],
            content_percentile=item["content_percentile"],
        ))

    if not args.dry_run:
        with (destination / "selected_tiles.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SELECTION_FIELDS)
            writer.writeheader()
            writer.writerows(asdict(row) for row in selected_rows)
    counts = {category: sum(row.category == category for row in selected_rows) for category in ("cell_rich", "low_cell", "negative_like")}
    summary = {
        "slide_id": folder.name,
        "source_directory": str(folder),
        "selected_directory": str(destination),
        "requested_tiles": args.per_wsi,
        "available_selection_target": target_count,
        "selected_tiles": len(selected_rows),
        "category_counts": counts,
        "eligible_tiles": len(scored),
        "seed": args.seed,
        "transfer": args.transfer,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not args.dry_run:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] {folder.name}: {counts}")
    return summary


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-root", type=Path, default=DEFAULT_TILES_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--per-wsi", type=int, default=DEFAULT_PER_WSI)
    parser.add_argument("--rich-fraction", type=float, default=DEFAULT_RICH_FRACTION)
    parser.add_argument("--negative-fraction", type=float, default=DEFAULT_NEGATIVE_FRACTION)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample", default=None, help="Process only the exact WSI folder name.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--transfer", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reselect", action="store_true", help="Recompute a completed selection; existing selected images are retained.")
    args = parser.parse_args(argv)
    if args.per_wsi <= 0 or args.workers <= 0:
        parser.error("--per-wsi and --workers must be positive")
    if not 0 < args.rich_fraction < 1 or not 0 <= args.negative_fraction < 1:
        parser.error("selection fractions must be between 0 and 1")
    if args.rich_fraction + args.negative_fraction >= 1:
        parser.error("rich plus negative fractions must leave room for low-cell tiles")
    args.tiles_root = args.tiles_root.expanduser().resolve()
    args.output = (args.output or args.tiles_root / OUTPUT_FOLDER_NAME).expanduser().resolve()
    return args


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        folders = discover_completed_wsi_folders(args.tiles_root, args.output)
        if args.sample:
            folders = [folder for folder in folders if folder.name == args.sample]
        if args.max_samples is not None:
            folders = folders[:args.max_samples]
        if not folders:
            raise FileNotFoundError("No matching completed WSI tile folders found.")
        print(f"[INFO] Found {len(folders)} completed WSI folder(s); output={args.output}")
        summaries = [select_for_wsi(folder, args.output, args) for folder in folders]
        if not args.dry_run:
            args.output.mkdir(parents=True, exist_ok=True)
            overall = {
                "configuration": {
                    "tiles_root": str(args.tiles_root), "per_wsi": args.per_wsi,
                    "rich_fraction": args.rich_fraction,
                    "negative_fraction": args.negative_fraction,
                    "workers": args.workers, "seed": args.seed,
                    "transfer": args.transfer,
                },
                "completed_wsi_count": len(summaries),
                "total_selected_tiles": sum(item["selected_tiles"] for item in summaries),
                "slides": summaries,
            }
            (args.output / "selection_summary.json").write_text(
                json.dumps(overall, indent=2), encoding="utf-8"
            )
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
