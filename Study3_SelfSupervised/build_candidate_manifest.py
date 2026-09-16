"""Build and audit a candidate specimen-level Study 3 data manifest.

Run directly from PyCharm. This reads existing split metadata and COCO files;
it does not move, copy, rename, or modify image data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_DATASET = (
    PROJECT_DIR
    / "SOLO_Supervised_RFDETR"
    / "Stat_Dataset40x"
    / "QA_40x-_20260901-135953"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "manifests"
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_from_filename(file_name: str) -> str:
    normalized = str(file_name).replace("\\", "/").strip("/")
    if "/" not in normalized:
        raise ValueError(f"Image path does not contain a specimen folder: {file_name}")
    return normalized.split("/", 1)[0]


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    summary_path = dataset_dir / "split_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing split summary: {summary_path}")

    summary = read_json(summary_path)
    samples = {split: list(summary["samples"][split]) for split in SPLITS}
    sample_sets = {split: set(values) for split, values in samples.items()}
    duplicates_within = {
        split: len(values) - len(sample_sets[split]) for split, values in samples.items()
    }
    overlaps = {
        "train_valid": sorted(sample_sets["train"] & sample_sets["valid"]),
        "train_test": sorted(sample_sets["train"] & sample_sets["test"]),
        "valid_test": sorted(sample_sets["valid"] & sample_sets["test"]),
    }
    if any(duplicates_within.values()):
        raise RuntimeError(f"Duplicate specimens within splits: {duplicates_within}")
    if any(overlaps.values()):
        raise RuntimeError(f"Specimen leakage between splits: {overlaps}")

    split_records = {}
    for split in SPLITS:
        coco_path = dataset_dir / split / "_annotations.coco.json"
        if not coco_path.is_file():
            raise FileNotFoundError(f"Missing COCO file: {coco_path}")
        coco = read_json(coco_path)
        filenames = [str(image["file_name"]) for image in coco.get("images", [])]
        coco_samples = {sample_from_filename(name) for name in filenames}
        unexpected = sorted(coco_samples - sample_sets[split])
        missing = sorted(sample_sets[split] - coco_samples)
        if unexpected or missing:
            raise RuntimeError(
                f"{split} COCO/specimen mismatch: unexpected={unexpected}, missing={missing}"
            )
        split_records[split] = {
            "specimens": samples[split],
            "n_specimens": len(samples[split]),
            "n_annotated_images": len(filenames),
            "n_annotations": len(coco.get("annotations", [])),
            "coco_file": str(coco_path),
            "coco_sha256": sha256(coco_path),
        }

    source_hash = sha256(summary_path)
    manifest = {
        "schema_version": 1,
        "status": "candidate_not_frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "specimen_boundary_for_study3_annotation_efficiency",
        "dataset_dir": str(dataset_dir),
        "dataset_name": summary.get("dataset_name"),
        "source_split_summary": str(summary_path),
        "source_split_summary_sha256": source_hash,
        "images_root_recorded_by_source": summary.get("images_root"),
        "split_strategy_recorded_by_source": summary.get("split_strategy"),
        "split_seed_recorded_by_source": summary.get("seed"),
        "target_classes": summary.get("target_classes", []),
        "audit": {
            "specimen_sets_disjoint": True,
            "duplicates_within_splits": duplicates_within,
            "overlaps": overlaps,
        },
        "splits": split_records,
        "ssl_eligibility": {
            "allowed_specimens": samples["train"],
            "forbidden_specimens": sorted(sample_sets["valid"] | sample_sets["test"]),
            "image_list_status": "not_yet_enumerated",
        },
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"candidate_40x_{source_hash[:12]}.json"
    if result_path.exists():
        previous = read_json(result_path)
        previous.pop("created_utc", None)
        comparison = dict(manifest)
        comparison.pop("created_utc", None)
        if previous != comparison:
            raise FileExistsError(f"Existing manifest differs: {result_path}")
        print(f"Candidate manifest already exists and matches: {result_path}")
        return 0

    result_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Candidate manifest audit passed")
    for split in SPLITS:
        record = split_records[split]
        print(
            f"{split}: specimens={record['n_specimens']} "
            f"annotated_images={record['n_annotated_images']} "
            f"annotations={record['n_annotations']}"
        )
    print(f"Saved manifest: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
