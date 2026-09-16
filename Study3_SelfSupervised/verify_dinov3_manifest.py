"""Validate and decode a manifest-backed DINOv3 image before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dinov3_manifest_dataset import MicroscopyManifest


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "manifests" / "local_ssl_subset_n8_6e908472b9d0.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = MicroscopyManifest(manifest=str(args.manifest), root=str(args.image_root))
    image, target = dataset[0]
    result = {
        "status": "passed",
        "manifest": str(dataset.manifest_path),
        "image_root": str(dataset.image_root),
        "number_of_images": len(dataset),
        "first_image": str(dataset.image_path(0)),
        "first_image_mode": image.mode,
        "first_image_size": list(image.size),
        "target_placeholder": target,
        "uses_biological_annotations": False,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

