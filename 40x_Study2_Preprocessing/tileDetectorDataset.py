"""Crop a specimen-level COCO dataset into overlapping native-pixel tiles."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

from PIL import Image


DATASETS = Path(__file__).resolve().parents[1] / "SOLO_Supervised_RFDETR" / "Stat_Dataset40x"
SOURCE = DATASETS / "QA-2025v1_TwoClass_OVR_V2_20261110-081112"
OUTPUT = DATASETS / "QA40x_TwoClass_Native704_20261110"
SPLITS = ("train", "valid", "test")
COCO_NAME = "_annotations.coco.json"


def offsets(length, size, max_stride):
    span = length - size
    if span < 0:
        raise ValueError(f"Image side {length} is smaller than crop size {size}")
    steps = math.ceil(span / max_stride)
    return [round(i * span / steps) for i in range(steps + 1)] if steps else [0]


def clip_box(bbox, left, top, size):
    x, y, width, height = bbox
    x1, y1 = max(0, x - left), max(0, y - top)
    x2, y2 = min(size, x + width - left), min(size, y + height - top)
    if x2 > x1 and y2 > y1:
        return [x1, y1, x2 - x1, y2 - y1]
    return None


def plan_split(coco, image_root, split_dir, size, max_stride, pad_only=False, resize_full=False):
    by_image = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    tiles, annotations, jobs = [], [], []
    covered, fully_visible, specimens = set(), set(), set()
    overlap_pixels = set()
    for parent in coco["images"]:
        parent_path = image_root / parent["file_name"]
        if not parent_path.is_file():
            raise FileNotFoundError(parent_path)
        specimen = parent_path.stem.split("_CAL10PATCH_", 1)[0]
        specimens.add(specimen)
        if resize_full:
            if min(parent["width"], parent["height"]) < size:
                raise ValueError(f"Cannot downsample {parent_path} to {size} without upscaling")
            xs = ys = [0]
        elif pad_only:
            if max(parent["width"], parent["height"]) > size:
                raise ValueError(f"Cannot pad {parent_path} to {size} without cropping")
            xs = ys = [0]
        else:
            xs = offsets(parent["width"], size, max_stride)
            ys = offsets(parent["height"], size, max_stride)
        for axis in (xs, ys):
            overlap_pixels.update(size - (b - a) for a, b in zip(axis, axis[1:]))
        crops = []
        for top in ys:
            for left in xs:
                name = f"{specimen}/{parent_path.stem}_TILE_X{left}_Y{top}.png"
                tile_id = len(tiles) + 1
                tiles.append(dict(id=tile_id, file_name=name, width=size, height=size,
                                  parent_image_id=parent["id"], parent_file_name=parent["file_name"],
                                  tile_x=left, tile_y=top))
                crops.append((left, top, split_dir / name))
                for source in by_image[parent["id"]]:
                    if resize_full:
                        x, y, w, h = source["bbox"]
                        sx, sy = size / parent["width"], size / parent["height"]
                        box = clip_box([x*sx, y*sy, w*sx, h*sy], 0, 0, size)
                        original_area = w * h * sx * sy
                    else:
                        box = clip_box(source["bbox"], left, top, size)
                        original_area = source["bbox"][2] * source["bbox"][3]
                    if box is None:
                        continue
                    covered.add(source["id"])
                    area = box[2] * box[3]
                    if math.isclose(area, original_area, rel_tol=1e-9):
                        fully_visible.add(source["id"])
                    annotations.append(dict(id=len(annotations) + 1, image_id=tile_id,
                                            category_id=source["category_id"], bbox=box, area=area,
                                            iscrowd=source.get("iscrowd", 0),
                                            parent_annotation_id=source["id"]))
        jobs.append((parent_path, (parent["width"], parent["height"]), crops))
    source_ids = {a["id"] for a in coco["annotations"]}
    if covered != source_ids:
        raise ValueError(f"{split_dir.name}: {len(source_ids - covered)} source boxes missed by grid")
    box_counts = Counter(a["category_id"] for a in annotations)
    summary = {
        "specimens": len(specimens), "parent_images": len(jobs), "tiles": len(tiles),
        "tiles_without_target_boxes": len(tiles) - len({a["image_id"] for a in annotations}),
        "parent_boxes": len(source_ids), "parent_boxes_fully_visible": len(fully_visible),
        "tile_boxes_by_class": {c["name"]: box_counts[c["id"]] for c in coco["categories"]},
        "actual_overlap_pixels": sorted(overlap_pixels),
    }
    return dict(images=tiles, annotations=annotations, categories=coco["categories"]), jobs, summary, specimens


def write_parent_tiles(job, size, sampling_size=None, resize_full=False):
    path, expected_size, crops = job
    with Image.open(path) as parent:
        if parent.size != expected_size:
            raise ValueError(f"COCO dimensions do not match image: {path}")
        parent.load()
        if resize_full:
            parent = parent.resize((size, size), Image.Resampling.LANCZOS)
        elif sampling_size is not None:
            # Match the 10x sampling grid, then restore the native dimensions
            # so the detector and COCO annotations use the original geometry.
            parent = parent.resize(sampling_size, Image.Resampling.LANCZOS).resize(
                expected_size, Image.Resampling.BICUBIC
            )
        for left, top, target in crops:
            parent.crop((left, top, left + size, top + size)).save(target, compress_level=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--tile-size", type=int, default=704)
    full_image = parser.add_mutually_exclusive_group()
    full_image.add_argument("--pad-only", action="store_true", help="Keep each whole patch and pad right/bottom")
    full_image.add_argument("--resize-full", action="store_true", help="Downsample each whole patch to the detector size")
    parser.add_argument("--sampling-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="With --pad-only, downsample to this grid and restore the native dimensions")
    parser.add_argument("--min-overlap", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    stride = int(args.tile_size * (1 - args.min_overlap))
    if args.tile_size < 1 or not 0 <= args.min_overlap < 1 or stride < 1 or args.workers < 1:
        parser.error("Tile size, stride and workers must be positive, and overlap in [0, 1).")
    if args.sampling_size is not None and (not args.pad_only or min(args.sampling_size) < 1):
        parser.error("--sampling-size requires --pad-only and positive dimensions.")
    image_root = args.image_root or Path(json.loads(
        (args.source / "split_summary.json").read_text(encoding="utf-8"))["images_root"])
    staging = args.output.with_name(args.output.name + ".partial")
    if args.output.exists() or staging.exists():
        raise FileExistsError(f"Choose an unused output path: {args.output} or {staging} exists")
    plans, specimens_seen = {}, set()
    for split in SPLITS:
        coco = json.loads((args.source / split / COCO_NAME).read_text(encoding="utf-8"))
        plans[split] = plan_split(coco, image_root, staging / split, args.tile_size, stride,
                                  args.pad_only, args.resize_full)
        if args.sampling_size is not None and any(
            any(sample >= native for sample, native in zip(args.sampling_size, dimensions))
            for _, dimensions, _ in plans[split][1]
        ):
            raise ValueError(f"{split}: sampling dimensions must be smaller than every native patch")
        specimens = plans[split][3]
        if specimens & specimens_seen:
            raise ValueError(f"Specimens appear in multiple splits: {specimens & specimens_seen}")
        specimens_seen.update(specimens)
    staging.mkdir(parents=True)
    report = {"source_dataset": str(args.source.resolve()), "tile_size": args.tile_size,
              "minimum_overlap_fraction": None if args.pad_only or args.resize_full else args.min_overlap,
              "pad_only": args.pad_only, "resized": args.resize_full or args.sampling_size is not None, "splits": {}}
    if args.resize_full:
        report["sampling"] = {"output_size": [args.tile_size, args.tile_size], "downsampling_filter": "Lanczos"}
    elif args.sampling_size is not None:
        report["sampling"] = {
            "intermediate_size": args.sampling_size,
            "downsampling_filter": "Lanczos",
            "restore_filter": "bicubic",
            "restored_to_native_dimensions_before_padding": True,
        }
    for split, (coco, jobs, summary, specimens) in plans.items():
        split_dir = staging / split
        split_dir.mkdir()
        for specimen in specimens:
            (split_dir / specimen).mkdir()
        print(f"[{split}] {summary['tiles']:,} tiles from {len(jobs)} parent patches", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, _ in enumerate(executor.map(
                lambda job: write_parent_tiles(job, args.tile_size, args.sampling_size, args.resize_full), jobs
            ), 1):
                if index % 100 == 0 or index == len(jobs):
                    print(f"[{split}] {index}/{len(jobs)} parent patches", flush=True)
        (split_dir / COCO_NAME).write_text(json.dumps(coco), encoding="utf-8")
        report["splits"][split] = summary
    (staging / "tiling_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    staging.rename(args.output)
    print(f"[DONE] {args.output}", flush=True)


if __name__ == "__main__":
    main()
