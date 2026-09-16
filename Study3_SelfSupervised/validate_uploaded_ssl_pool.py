"""Audit an uploaded SSL image tree against its frozen inventory manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "manifests" / "ssl_pool_40x_9d8cb0d9ec7b.csv"
DEFAULT_SUMMARY = DEFAULT_MANIFEST.with_suffix(".summary.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--check-sizes", action="store_true")
    parser.add_argument("--max-reported-errors", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve()
    image_root = args.image_root.expanduser().resolve()
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    missing = []
    size_mismatches = []
    unsafe = []
    duplicates = []
    seen = set()
    specimens = set()
    manifest_bytes = 0
    present_bytes = 0
    rows = 0
    with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            relative_text = row["relative_path"].strip()
            relative = Path(relative_text)
            expected_size = int(row["size_bytes"])
            manifest_bytes += expected_size
            specimens.add(row["specimen"])
            if relative.is_absolute() or ".." in relative.parts:
                unsafe.append(relative_text)
                continue
            if relative_text in seen:
                duplicates.append(relative_text)
            seen.add(relative_text)
            path = image_root / relative
            try:
                actual_size = path.stat().st_size
                present_bytes += actual_size
                if args.check_sizes and actual_size != expected_size:
                    size_mismatches.append(
                        {"path": relative_text, "expected": expected_size, "actual": actual_size}
                    )
            except FileNotFoundError:
                missing.append(relative_text)

    expected_hash = source_summary.get("inventory_csv_sha256")
    actual_hash = sha256(manifest)
    expected_rows = int(source_summary["n_total_ssl_eligible_images"])
    expected_specimens = int(source_summary["n_training_specimens"])
    expected_bytes = int(source_summary["total_size_bytes"])
    passed = all(
        (
            rows == expected_rows,
            len(specimens) == expected_specimens,
            manifest_bytes == expected_bytes,
            actual_hash == expected_hash,
            not missing,
            not unsafe,
            not duplicates,
            not size_mismatches,
        )
    )
    limit = max(0, args.max_reported_errors)
    result = {
        "status": "passed" if passed else "failed",
        "manifest": str(manifest),
        "manifest_sha256": actual_hash,
        "manifest_hash_matches_summary": actual_hash == expected_hash,
        "image_root": str(image_root),
        "rows": rows,
        "expected_rows": expected_rows,
        "specimens": len(specimens),
        "expected_specimens": expected_specimens,
        "manifest_bytes": manifest_bytes,
        "expected_bytes": expected_bytes,
        "present_bytes": present_bytes,
        "check_sizes": args.check_sizes,
        "missing_count": len(missing),
        "size_mismatch_count": len(size_mismatches),
        "unsafe_path_count": len(unsafe),
        "duplicate_count": len(duplicates),
        "missing_examples": missing[:limit],
        "size_mismatch_examples": size_mismatches[:limit],
        "unsafe_path_examples": unsafe[:limit],
        "duplicate_examples": duplicates[:limit],
    }
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

