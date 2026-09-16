"""Auditable launcher for official DINOv3 self-supervised pretraining.

This wrapper deliberately contains no SSL implementation. It validates and
records the inputs, then delegates training to a separately installed checkout
of Meta's official DINOv3 repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch official DINOv3 SSL training with recorded provenance."
    )
    parser.add_argument("--dinov3-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset-spec",
        required=True,
        help="Official DINOv3 dataset specification; must describe training data only.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch training. Without this flag the script performs a dry run.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Optional official DINOv3 configuration overrides.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.dinov3_repo.resolve()
    config = args.config.resolve()
    output_dir = args.output_dir.resolve()
    train_script = repo / "dinov3" / "train" / "train.py"

    if not train_script.is_file():
        raise FileNotFoundError(f"Official DINOv3 training script not found: {train_script}")
    if not config.is_file():
        raise FileNotFoundError(f"DINOv3 configuration not found: {config}")
    try:
        config.relative_to(repo)
    except ValueError as exc:
        raise ValueError("--config must be inside the specified DINOv3 repository") from exc
    if not args.dataset_spec.strip():
        raise ValueError("--dataset-spec cannot be empty")

    # An existing checkpoint directory can trigger implicit resume in upstream
    # code. We pass --no-resume and reject non-empty output directories so every
    # experimental run has an unambiguous starting state.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(train_script),
        "--config-file",
        str(config),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--no-resume",
        f"train.dataset_path={args.dataset_spec}",
        *args.overrides,
    ]
    manifest = {
        "study": "quality_assessment_annotation_efficiency",
        "stage": "own_data_ssl_pretraining",
        "initialization": "random_no_external_checkpoint",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": not args.execute,
        "python": sys.version,
        "platform": platform.platform(),
        "dinov3_repo": str(repo),
        "train_script": str(train_script),
        "config": str(config),
        "config_sha256": sha256(config),
        "dataset_spec": args.dataset_spec,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "overrides": args.overrides,
        "command": command,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("DINOv3 SSL command:")
    print(shlex.join(command))
    print(f"Run manifest: {manifest_path}")
    if not args.execute:
        print("Dry run only. Add --execute after reviewing the manifest and config.")
        return 0

    if platform.system() != "Linux":
        print(
            "WARNING: Meta documents official DINOv3 training for Linux. "
            "This local execution is an unsupported compatibility test."
        )
    completed = subprocess.run(command, cwd=repo, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
