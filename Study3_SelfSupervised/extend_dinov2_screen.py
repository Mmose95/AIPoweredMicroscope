"""Continue six teacher screens to epoch 40 and train a random-encoder control."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from probe_dinov2_teachers import prepare, read

STOCK_SHA = "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38"


def plan(args):
    teachers = [("original20", args.work / "ssl_dinov2_small16_scratch/teacher_epoch_020.pth")]
    teachers += [(f"long{e}", args.work / f"ssl_dinov2_small16_scratch_200/teacher_epoch_{e:03d}.pth")
                 for e in (20, 50, 100, 150, 200)]
    jobs = []
    for label, teacher in teachers + [("random", None)]:
        output = args.output / label
        if (output / "complete.json").is_file():
            record = read(output / "complete.json")
            if record["epochs"] != 40 or record["teacher"] != (str(teacher) if teacher else None):
                raise ValueError(f"Completed output has different settings: {output}")
            continue
        if output.exists():
            raise FileExistsError(f"Interrupted output exists; preserve it and choose a new --output: {output}")
        command = [sys.executable, "-u", str(Path(__file__).with_name("probe_dinov2_teachers.py")),
                   "--worker", "--full-finetune", "--epochs", "40", "--weights", str(args.weights),
                   "--dataset", str(args.work / "full_native2528"), "--output", str(output)]
        if teacher:
            source = args.source / label
            provenance = read(source / "complete.json")
            sha = hashlib.sha256(teacher.read_bytes()).hexdigest()
            if (provenance["epochs"] != 10 or provenance["frozen_encoder"]
                    or provenance["detector"] != "large" or provenance["stock_sha256"] != STOCK_SHA
                    or provenance["teacher_sha256"] != sha):
                raise ValueError(f"Unexpected source run provenance: {source}")
            for name in ("last.ckpt", "initialization.pth", "metrics.csv"):
                if not (source / name).is_file():
                    raise FileNotFoundError(source / name)
            command += ["--teacher", str(teacher), "--resume", str(source / "last.ckpt"),
                        "--initialization", str(source / "initialization.pth"),
                        "--history", str(source / "metrics.csv")]
        jobs.append((label, command))
    return jobs


def main(args):
    import torch
    if torch.cuda.device_count() < 7:
        raise RuntimeError("Seven visible GPUs are required: six teachers plus random control")
    if hashlib.sha256(args.weights.read_bytes()).hexdigest() != STOCK_SHA:
        raise ValueError("Published RF-DETR Large checkpoint checksum mismatch")
    jobs = plan(args)
    # Validate checkpoint training state before launching any jobs.
    for label, _ in jobs:
        if label == "random":
            continue
        checkpoint = torch.load(args.source / label / "last.ckpt", map_location="cpu", weights_only=False)
        if checkpoint["epoch"] != 9 or not checkpoint.get("optimizer_states") or not checkpoint.get("lr_schedulers"):
            raise ValueError(f"Expected resumable epoch-10 checkpoint: {label}")
        del checkpoint
    args.full_finetune = True
    prepare(args)
    args.output.mkdir(parents=True, exist_ok=True)
    running = []
    try:
        # Preserve CUDA_VISIBLE_DEVICES UUIDs or indices if the job defines them.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = visible.split(",") if visible else [str(i) for i in range(torch.cuda.device_count())]
        for gpu, (label, command) in zip(devices, jobs):
            log_path = args.output / f"{label}.log"
            log = log_path.open("w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="2", NO_ALBUMENTATIONS_UPDATE="1")
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((label, process, log))
            print(f"GPU {gpu}: {label}: {log_path}", flush=True)
        failed = []
        while running:
            for item in running[:]:
                label, process, log = item
                status = process.poll()
                if status is not None:
                    log.close()
                    running.remove(item)
                    print(f"{label}: exit {status}", flush=True)
                    if status:
                        failed.append(label)
            if running:
                time.sleep(5)
    except BaseException:
        for _, process, log in running:
            process.terminate()
            process.wait()
            log.close()
        raise
    rows = []
    for output in sorted(args.output.iterdir()):
        if not output.is_dir() or not (output / "complete.json").is_file():
            continue
        with (output / "metrics.csv").open(newline="") as stream:
            metrics = [r for r in csv.DictReader(stream) if r.get("val/ema_mAP_50_95")]
        if len({r["epoch"] for r in metrics}) != 40:
            raise ValueError(f"Expected 40 validation epochs: {output}")
        best = max(metrics, key=lambda r: float(r["val/ema_mAP_50_95"]))
        rows.append(dict(teacher=output.name, best_epoch=int(best["epoch"]) + 1,
                         ema_map=float(best["val/ema_mAP_50_95"])))
    rows.sort(key=lambda r: r["ema_map"], reverse=True)
    (args.output / "leaderboard.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2), flush=True)
    if failed:
        raise RuntimeError(f"Runs failed; see individual logs: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    main(parser.parse_args())
