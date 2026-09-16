"""Build a small, deterministic, specimen-balanced local SSL subset manifest.

No images are copied. Selection spans each specimen's white-background spectrum
using acquisition-time tile statistics, without using biological annotations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_DIR = SCRIPT_DIR / "manifests"


def latest_inventory_summary() -> Path:
    candidates = sorted(MANIFEST_DIR.glob("ssl_pool_40x_*.summary.json"))
    return candidates[-1] if candidates else MANIFEST_DIR / "ssl_pool_MISSING.summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-summary", type=Path, default=latest_inventory_summary())
    parser.add_argument("--images-per-specimen", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=MANIFEST_DIR)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count > length:
        raise ValueError(f"Cannot select {count} unique items from {length}")
    if count == 1:
        return [length // 2]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def main() -> int:
    args = parse_args()
    if args.images_per_specimen < 1:
        raise ValueError("--images-per-specimen must be positive")
    summary_path = args.inventory_summary.resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Inventory summary not found: {summary_path}")
    summary = read_json(summary_path)
    images_root = Path(summary["images_root"])
    if not images_root.is_dir():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    selected = []
    for specimen in summary["per_specimen"]:
        tiles_csv = images_root / specimen / "tiles.csv"
        if not tiles_csv.is_file():
            raise FileNotFoundError(f"Tile metadata not found: {tiles_csv}")
        with tiles_csv.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        eligible = []
        for row in rows:
            image_path = images_root / specimen / row["file_name"]
            if not image_path.is_file() or not row.get("white_fraction"):
                continue
            eligible.append(
                {
                    "specimen": specimen,
                    "relative_path": image_path.relative_to(images_root).as_posix(),
                    "local_absolute_path": str(image_path.resolve()),
                    "white_fraction": float(row["white_fraction"]),
                    "mean_intensity": float(row["mean_intensity"]),
                    "row": int(row["row"]),
                    "col": int(row["col"]),
                }
            )
        eligible.sort(key=lambda item: (item["white_fraction"], item["relative_path"]))
        for rank, index in enumerate(
            evenly_spaced_indices(len(eligible), args.images_per_specimen)
        ):
            item = dict(eligible[index])
            item["within_specimen_selection_rank"] = rank
            selected.append(item)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(summary_path)
    stem = f"local_ssl_subset_n{args.images_per_specimen}_{source_hash[:12]}"
    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    subset_summary = {
        "schema_version": 1,
        "purpose": "local_real_image_ssl_integration_test",
        "research_training_eligible": False,
        "selection_method": "even_quantiles_of_white_fraction_within_each_specimen",
        "uses_biological_annotations_for_selection": False,
        "source_inventory_summary": str(summary_path),
        "source_inventory_summary_sha256": source_hash,
        "images_root": str(images_root.resolve()),
        "n_specimens": len(summary["per_specimen"]),
        "images_per_specimen": args.images_per_specimen,
        "n_images": len(selected),
        "subset_csv": str(csv_path),
        "subset_csv_sha256": sha256(csv_path),
    }
    json_path = output_dir / f"{stem}.summary.json"
    json_path.write_text(json.dumps(subset_summary, indent=2) + "\n", encoding="utf-8")
    print("Local SSL subset created")
    print(f"specimens: {subset_summary['n_specimens']}")
    print(f"images per specimen: {args.images_per_specimen}")
    print(f"total images: {len(selected)}")
    print(f"CSV: {csv_path}")
    print(f"summary: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
