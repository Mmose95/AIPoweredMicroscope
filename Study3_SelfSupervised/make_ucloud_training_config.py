"""Generate a reviewable full-pool DINOv3 config after GPU allocation is known."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = SCRIPT_DIR / "configs" / "dinov3_vits16_wsl_pilot.yaml"
DEFAULT_MANIFEST = SCRIPT_DIR / "manifests" / "ssl_pool_40x_9d8cb0d9ec7b.csv"


def count_images(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--batch-size-per-gpu", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--warmup-epochs", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("gpus", "batch_size_per_gpu", "epochs"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.warmup_epochs <= args.epochs:
        raise ValueError("--warmup-epochs must be between zero and --epochs")

    image_count = count_images(args.manifest)
    global_batch = args.gpus * args.batch_size_per_gpu
    iterations_per_epoch = math.ceil(image_count / global_batch)
    cfg = OmegaConf.load(args.base_config)
    cfg.train.batch_size_per_gpu = args.batch_size_per_gpu
    cfg.train.num_workers = args.num_workers
    cfg.train.OFFICIAL_EPOCH_LENGTH = iterations_per_epoch
    cfg.optim.epochs = args.epochs
    cfg.optim.warmup_epochs = args.warmup_epochs
    cfg.optim.freeze_last_layer_epochs = min(1, args.warmup_epochs)
    cfg.teacher.warmup_teacher_temp_epochs = args.warmup_epochs
    cfg.crops.local_crops_number = 8
    cfg.dino.head_n_prototypes = 65536
    cfg.ibot.head_n_prototypes = 65536
    cfg.evaluation.eval_period_iterations = iterations_per_epoch
    cfg.checkpointing.period = iterations_per_epoch
    cfg.checkpointing.max_to_keep = 3

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output)
    print(f"Saved: {output}")
    print(f"Images: {image_count:,}")
    print(f"Global batch: {global_batch:,}")
    print(f"Iterations per nominal epoch: {iterations_per_epoch:,}")
    print(f"Total planned iterations: {iterations_per_epoch * args.epochs:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

