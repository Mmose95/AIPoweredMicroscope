"""Screen six DINOv2 teachers, one detector per GPU; optional full Large fine-tuning."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from collections import defaultdict

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def worker(args):
    import torch
    from rfdetr import RFDETRSmall, RFDETRLarge
    from dinov2_stock_initialization import write_initialization

    random.seed(0)
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)
    args.output.mkdir(parents=True)
    initialization = args.initialization or args.output / "initialization.pth"
    if args.initialization is None:
        write_initialization(args.weights, initialization, args.teacher)
    detector = RFDETRLarge if args.full_finetune else RFDETRSmall
    model = detector(pretrain_weights=str(initialization), resolution=2528, patch_size=16,
                     positional_encoding_size=158, num_queries=300,
                     gradient_checkpointing=args.full_finetune, freeze_encoder=not args.full_finetune)
    provenance = dict(teacher=str(args.teacher) if args.teacher else None,
                      teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest() if args.teacher else None,
                      stock_weights=str(args.weights), stock_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                      epochs=args.epochs, seed=0, resolution=2528, frozen_encoder=not args.full_finetune,
                      detector="large" if args.full_finetune else "small", dataset=str(args.dataset))
    provenance["resume_checkpoint"] = str(args.resume) if args.resume else None
    provenance["backbone_initialization"] = "own_ssl" if args.teacher else "random"
    if args.history:
        (args.output / "metrics_first10.csv").write_bytes(args.history.read_bytes())
    (args.output / "probe.json").write_text(json.dumps(provenance, indent=2))
    model.train(dataset_dir=str(args.dataset), output_dir=str(args.output),
                class_names=["Leucocyte", "Squamous Epithelial Cell"], epochs=args.epochs,
                batch_size=1, grad_accum_steps=16, lr=5e-5, weight_decay=.0007,
                num_workers=4 if args.full_finetune else 2, checkpoint_interval=5, seed=0, multi_scale=False,
                scale_jitter=False, do_random_resize_via_padding=False, early_stopping=False,
                run_test=False, progress_bar="tqdm", resume=str(args.resume) if args.resume else None)
    encoder = model.model.model.backbone[0].encoder
    if not args.full_finetune and any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("The probe encoder must stay frozen")
    if args.full_finetune and not any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("The screening encoder must be trainable")
    if args.history:
        merge_metrics(args.output / "metrics_first10.csv", args.output / "metrics.csv")
    (args.output / "complete.json").write_text(json.dumps(provenance, indent=2))


def merge_metrics(history, destination):
    """CSVLogger replaces existing logs; retain the first ten epochs explicitly."""
    with history.open(newline="") as stream:
        previous = list(csv.DictReader(stream))
    with destination.open(newline="") as stream:
        current = list(csv.DictReader(stream))
    fields = list(dict.fromkeys(key for row in previous + current for key in row))
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(previous + current)


def prepare(args):
    native = args.work / "full_native2528"
    if args.full_finetune:
        for split in ("train", "valid"):
            coco = read(native / split / "_annotations.coco.json")
            for image in coco["images"]:
                if not (native / split / image["file_name"]).is_file():
                    raise FileNotFoundError(native / split / image["file_name"])
            print(f"[{split}] {len(coco['images'])} full training-size patches", flush=True)
        return native
    train = read(native / "train/_annotations.coco.json")
    valid = read(native / "valid/_annotations.coco.json")
    grouped = defaultdict(list)
    for image in train["images"]:
        grouped[image["file_name"].split("/")[0]].append(image)
    rng = random.Random(0)
    for images in grouped.values():
        rng.shuffle(images)
    selected = []
    while len(selected) < 150:
        for specimen in sorted(grouped):
            if grouped[specimen] and len(selected) < 150:
                selected.append(grouped[specimen].pop())
        if not any(grouped.values()) and len(selected) < 150:
            raise ValueError("Fewer than 150 training patches")
    ids = {image["id"] for image in selected}
    train["images"] = selected
    train["annotations"] = [a for a in train["annotations"] if a["image_id"] in ids]
    dataset = args.output / "dataset"
    for split, coco in (("train", train), ("valid", valid)):
        for image in coco["images"]:
            source = native / split / image["file_name"]
            if not source.is_file():
                raise FileNotFoundError(source)
            image["file_name"] = str(source.resolve())
        destination = dataset / split / "_annotations.coco.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(coco)
        if destination.exists() and destination.read_text() != content:
            raise ValueError(f"Existing probe dataset differs: {destination}")
        destination.write_text(content)
    return dataset


def main(args):
    if args.worker:
        worker(args)
        return
    import torch
    if torch.cuda.device_count() < 6:
        raise RuntimeError("This launcher needs six visible GPUs")
    teachers = [("original20", args.work / "ssl_dinov2_small16_scratch/teacher_epoch_020.pth")]
    teachers += [(f"long{epoch}", args.work / f"ssl_dinov2_small16_scratch_200/teacher_epoch_{epoch:03d}.pth")
                 for epoch in (20, 50, 100, 150, 200)]
    for label, _ in teachers:
        output = args.output / label
        if output.exists() and not (output / "complete.json").is_file():
            raise FileExistsError(f"Interrupted probe folder exists. Use a fresh --output: {output}")
    for _, teacher in teachers:
        payload = torch.load(teacher, map_location="cpu", weights_only=True)
        if "teacher" not in payload:
            raise ValueError(f"Teacher export missing: {teacher}")
    del payload
    dataset = prepare(args)
    # Resolve the published Small checkpoint once before starting concurrent workers.
    if args.full_finetune:
        if args.weights is None or not args.weights.is_file():
            raise FileNotFoundError("Pass --weights with the published RF-DETR Large checkpoint")
        weights = args.weights.resolve()
        expected = "0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != expected:
            raise ValueError("RF-DETR Large stock checkpoint checksum mismatch")
    else:
        from rfdetr import RFDETRSmall
        stock = RFDETRSmall()
        weights = Path(stock.model_config.pretrain_weights).resolve()
        del stock
    args.output.mkdir(parents=True, exist_ok=True)
    processes = []
    for gpu, (label, teacher) in enumerate(teachers):
        output = args.output / label
        if (output / "complete.json").is_file():
            print(f"Already complete: {label}", flush=True)
            continue
        if output.exists():
            raise FileExistsError(f"Interrupted probe folder exists. Use a fresh --output: {output}")
        log = (args.output / f"{label}.log").open("w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", NO_ALBUMENTATIONS_UPDATE="1")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                   "--teacher", str(teacher), "--weights", str(weights), "--dataset", str(dataset),
                   "--output", str(output), "--epochs", str(args.epochs)]
        if args.full_finetune:
            command.append("--full-finetune")
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((label, process, log))
        print(f"GPU {gpu}: {label} → {args.output / (label + '.log')}", flush=True)
    pending = processes[:]
    failed = []
    while pending:
        for item in pending[:]:
            label, process, log = item
            status = process.poll()
            if status is not None:
                log.close()
                pending.remove(item)
                print(f"{label}: exit {status}", flush=True)
                if status:
                    failed.append(label)
        if pending:
            time.sleep(5)
    rows = []
    for label, teacher in teachers:
        metrics = args.output / label / "metrics.csv"
        if not (args.output / label / "complete.json").exists():
            continue
        with metrics.open(newline="") as stream:
            epochs = [r for r in csv.DictReader(stream) if r.get("val/ema_mAP_50_95")]
        best = max(epochs, key=lambda r: float(r["val/ema_mAP_50_95"]))
        rows.append(dict(teacher=label, checkpoint=str(teacher), best_epoch=int(best["epoch"])+1,
                         ema_map=float(best["val/ema_mAP_50_95"])))
    rows.sort(key=lambda row: row["ema_map"], reverse=True)
    (args.output / "leaderboard.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2), flush=True)
    if failed:
        raise RuntimeError(f"Probes failed, see logs: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--full-finetune", action="store_true",
                        help="Large, trainable backbone, all training patches; use a separate output")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initialization", type=Path)
    parser.add_argument("--history", type=Path)
    options = parser.parse_args()
    if not options.worker and options.work is None:
        parser.error("--work is required for the six-GPU launcher")
    main(options)
