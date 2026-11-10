"""Build an SSL inventory from the original level-0 tiles of training specimens."""
import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import re

from enumerate_ssl_pool import SUPPORTED_EXTENSIONS


def specimen(name):
    match = re.match(r"^sample[ _]*(\d+)(?=$|[ _.\-])", name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot identify specimen: {name}")
    return int(match.group(1))


def split_specimens(dataset):
    splits = {}
    seen = set()
    for name in ("train", "valid", "test"):
        coco = json.loads((dataset / name / "_annotations.coco.json").read_text(encoding="utf-8"))
        ids = {specimen(PurePosixPath(i["file_name"]).name) for i in coco["images"]}
        if not ids or ids & seen:
            raise ValueError(f"Empty or overlapping specimen split: {name}")
        splits[name] = ids
        seen.update(ids)
    return splits


def build(dataset, image_root, output):
    allowed = split_specimens(dataset)["train"]
    folders = {}
    for folder in image_root.iterdir():
        if not folder.is_dir() or not re.match(r"^sample[ _]*\d", folder.name, re.IGNORECASE):
            continue
        sid = specimen(folder.name)
        if sid in allowed:
            if sid in folders:
                raise ValueError(f"More than one folder for training specimen {sid}")
            folders[sid] = folder
    if allowed - folders.keys():
        raise FileNotFoundError(f"Missing SSL training specimen folders: {sorted(allowed - folders.keys())}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial")
    count = 0
    with partial.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("specimen", "relative_path", "extension", "size_bytes"))
        for sid, folder in sorted(folders.items()):
            if folder.is_symlink():
                raise ValueError(f"Refusing a symlinked SSL specimen folder: {folder}")
            paths = sorted(p for p in folder.iterdir()
                           if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
            if not paths:
                raise ValueError(f"No SSL tiles in {folder}")
            for path in paths:
                size = path.stat().st_size
                if specimen(path.name) != sid or path.is_symlink() or size == 0:
                    raise ValueError(f"Invalid SSL tile for specimen {sid}: {path}")
                writer.writerow((folder.name, path.relative_to(image_root).as_posix(),
                                 path.suffix.lower(), size))
                count += 1
    if output.exists():
        matches = partial.read_bytes() == output.read_bytes()
        partial.unlink()
        if not matches:
            raise ValueError(f"SSL inventory has changed. Use a new run directory: {output}")
    else:
        partial.rename(output)
    print(f"SSL manifest: {count:,} tiles from {len(allowed)} training specimens: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.dataset, args.image_root, args.output)
