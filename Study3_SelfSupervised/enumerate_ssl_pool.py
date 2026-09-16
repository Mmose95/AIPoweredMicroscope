"""Enumerate all SSL-eligible images from candidate training specimens.

Run directly from PyCharm. The script reads image metadata only and never
copies, moves, renames, or edits source images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_DIR = SCRIPT_DIR / "manifests"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def latest_candidate_manifest() -> Path:
    candidates = sorted(DEFAULT_MANIFEST_DIR.glob("candidate_40x_*.json"))
    if not candidates:
        return DEFAULT_MANIFEST_DIR / "candidate_40x_MISSING.json"
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, default=latest_candidate_manifest())
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_relative(path: str) -> str:
    return str(path).replace("\\", "/").strip("/")


def main() -> int:
    args = parse_args()
    candidate_path = args.candidate_manifest.resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate manifest not found: {candidate_path}")
    candidate = read_json(candidate_path)

    recorded_root = candidate.get("images_root_recorded_by_source")
    images_root = (args.images_root or Path(recorded_root)).resolve()
    if not images_root.is_dir():
        raise FileNotFoundError(f"Full tiled image root not found: {images_root}")

    allowed = list(candidate["ssl_eligibility"]["allowed_specimens"])
    forbidden = set(candidate["ssl_eligibility"]["forbidden_specimens"])
    if set(allowed) & forbidden:
        raise RuntimeError("Candidate manifest allows forbidden specimens")

    train_coco_path = Path(candidate["splits"]["train"]["coco_file"])
    train_coco = read_json(train_coco_path)
    annotated = {
        normalize_relative(image["file_name"]) for image in train_coco.get("images", [])
    }

    rows = []
    missing_specimen_folders = []
    counts_by_specimen = Counter()
    annotated_by_specimen = Counter()
    bytes_by_specimen = Counter()
    for specimen in allowed:
        specimen_dir = images_root / specimen
        if not specimen_dir.is_dir():
            missing_specimen_folders.append(specimen)
            continue
        for path in sorted(specimen_dir.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            relative = normalize_relative(path.relative_to(images_root))
            is_annotated = relative in annotated
            size_bytes = path.stat().st_size
            rows.append(
                {
                    "specimen": specimen,
                    "relative_path": relative,
                    "extension": path.suffix.lower(),
                    "size_bytes": size_bytes,
                    "is_annotated_training_image": int(is_annotated),
                }
            )
            counts_by_specimen[specimen] += 1
            bytes_by_specimen[specimen] += size_bytes
            if is_annotated:
                annotated_by_specimen[specimen] += 1

    if missing_specimen_folders:
        raise FileNotFoundError(f"Missing training specimen folders: {missing_specimen_folders}")
    found_relatives = {row["relative_path"] for row in rows}
    missing_annotated = sorted(annotated - found_relatives)
    if missing_annotated:
        raise FileNotFoundError(
            f"{len(missing_annotated)} annotated training images are absent from the full tile root; "
            f"first examples: {missing_annotated[:10]}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_hash = sha256(candidate_path)
    inventory_path = output_dir / f"ssl_pool_40x_{candidate_hash[:12]}.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    annotated_count = sum(row["is_annotated_training_image"] for row in rows)
    summary = {
        "schema_version": 1,
        "status": "candidate_not_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_type": "full_level0_tiles_from_training_specimens",
        "candidate_manifest": str(candidate_path),
        "candidate_manifest_sha256": candidate_hash,
        "images_root": str(images_root),
        "n_training_specimens": len(allowed),
        "n_total_ssl_eligible_images": total,
        "n_annotated_training_images": annotated_count,
        "n_additional_unannotated_images": total - annotated_count,
        "annotated_fraction": annotated_count / total if total else None,
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "forbidden_specimens_scanned": 0,
        "inventory_csv": str(inventory_path),
        "inventory_csv_sha256": sha256(inventory_path),
        "per_specimen": {
            specimen: {
                "total_images": counts_by_specimen[specimen],
                "annotated_images": annotated_by_specimen[specimen],
                "additional_unannotated_images": (
                    counts_by_specimen[specimen] - annotated_by_specimen[specimen]
                ),
                "size_bytes": bytes_by_specimen[specimen],
            }
            for specimen in allowed
        },
    }
    summary_path = inventory_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("SSL pool inventory passed")
    print(f"training specimens: {len(allowed)}")
    print(f"total eligible images: {total}")
    print(f"annotated training images: {annotated_count}")
    print(f"additional unannotated images: {total - annotated_count}")
    print(f"annotated fraction: {annotated_count / total:.4%}")
    print(f"inventory: {inventory_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
