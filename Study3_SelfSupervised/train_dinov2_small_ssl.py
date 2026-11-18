"""DINOv2 SSL for the RF-DETR Large Small/16 encoder."""
import argparse
import csv
from functools import partial
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather
from torch.nn.parallel import DistributedDataParallel
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from dinov2.data.augmentations import DataAugmentationDINO
from dinov2.data.collate import collate_data_and_cast
from dinov2.data.masking import MaskingGenerator
from dinov2.layers.dino_head import DINOHead
from dinov2.loss import DINOLoss, KoLeoLoss, iBOTPatchLoss
from rfdetr.models.backbone.dinov2 import DinoV2

PREFIX = "backbone.0.encoder.encoder."


class ManifestImages(Dataset):
    def __init__(self, manifest, root, local_crops=8):
        with manifest.open(encoding="utf-8", newline="") as handle:
            self.paths = [root / row["relative_path"] for row in csv.DictReader(handle)]
        if len(self.paths) < 2:
            raise ValueError("SSL needs at least two training images")
        self.transform = DataAugmentationDINO((0.32, 1.0), (0.05, 0.32), local_crops, 224, 96)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), ()


def make_encoder():
    # Full attention for SSL crops. Windowing in RF-DETR changes attention
    # grouping at inference, while the encoder parameter names/shapes match.
    return DinoV2(shape=(2528, 2528), size="small", use_registers=False,
                  out_feature_indexes=[3, 6, 9, 12], load_dinov2_weights=False,
                  patch_size=16, positional_encoding_size=158, num_windows=1).encoder


def stock_encoder_state(source, encoder):
    from rfdetr.models.weights import interpolate_position_embeddings

    state = {name: value for name, value in source["model"].items() if name.startswith(PREFIX)}
    interpolate_position_embeddings(state, encoder.config.image_size // encoder.config.patch_size)
    return {name.removeprefix(PREFIX): value for name, value in state.items()}


def tokens(encoder, images, masks=None):
    embeddings = encoder.embeddings(images, bool_masked_pos=masks)
    encoded = encoder.encoder(embeddings, return_dict=False)[0]
    return encoder.layernorm(encoded)


class SSLNetwork(nn.ModuleDict):
    def forward(self, global_views, local_views, mask):
        global_tokens = tokens(self["backbone"], global_views, mask)
        local_logits = None
        if local_views.shape[0]:
            local_tokens = tokens(self["backbone"], local_views)
            local_logits = self["dino"](local_tokens[:, 0]).float()
        return (self["dino"](global_tokens[:, 0]).float(),
                local_logits,
                self["ibot"](global_tokens[:, 1:][mask]).float(),
                global_tokens[:, 0].float())


def ssl_network(encoder):
    return SSLNetwork({
        "backbone": encoder, "dino": DINOHead(384, 65536), "ibot": DINOHead(384, 65536),
    })


def cross_view_dino_loss(dino, student_global, student_local, teacher_prob, local_count):
    s0, s1 = student_global.chunk(2)
    t0, t1 = teacher_prob.chunk(2)
    loss = dino([s0], [t1]) + dino([s1], [t0])
    if local_count:
        loss = loss + dino(student_local.chunk(local_count), [t0, t1])
    return loss / (2 + 2 * local_count)


def train(args):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.batch_size % world or args.batch_size // world < 2:
        raise ValueError("Global batch size must divide evenly across GPUs with at least two images per GPU")
    if rank == 0 and (args.output / "config.json").exists() and not args.resume:
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda", local_rank)
    dataset = ManifestImages(args.manifest, args.images, args.local_crops)
    collate = partial(collate_data_and_cast, mask_ratio_tuple=(0.1, 0.5),
                      mask_probability=0.5, dtype=torch.float32, n_tokens=196,
                      mask_generator=MaskingGenerator((14, 14), max_num_patches=98))
    sampler = DistributedSampler(dataset, shuffle=True, seed=0) if world > 1 else None
    loader = DataLoader(dataset, batch_size=args.batch_size // world, shuffle=sampler is None,
                        sampler=sampler, num_workers=args.workers,
                        collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)
    encoder = make_encoder()
    if args.initial_checkpoint:
        source = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        encoder.load_state_dict(stock_encoder_state(source, encoder), strict=True)
    student = ssl_network(encoder)
    teacher = ssl_network(make_encoder()).requires_grad_(False).eval()
    teacher.load_state_dict(student.state_dict())
    student.to(device)
    teacher.to(device)
    parallel_student = DistributedDataParallel(student, device_ids=[local_rank],
                                               gradient_as_bucket_view=True) if world > 1 else student
    random.seed(rank)
    torch.manual_seed(rank)
    dino = DINOLoss(65536).to(device)
    ibot = iBOTPatchLoss(65536).to(device)
    koleo = KoLeoLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    steps = len(loader) * args.epochs
    warmup = len(loader) * args.warmup
    record = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    record.update(images_in_manifest=len(dataset), steps_per_epoch=len(loader),
                  torch=str(torch.__version__), rfdetr=version("rfdetr"), world_size=world)
    start_epoch = 0
    if args.resume:
        saved = torch.load(args.output / "last.pt", map_location="cpu", weights_only=True)
        saved["config"].setdefault("local_crops", 8)
        if saved["config"] != {**record, "resume": False}:
            raise ValueError("Resume with the original configuration and GPU count")
        student.load_state_dict(saved["student"])
        teacher.load_state_dict(saved["teacher"])
        optimizer.load_state_dict(saved["optimizer"])
        dino.load_state_dict(saved["dino"])
        ibot.load_state_dict(saved["ibot"])
        start_epoch = saved["epoch"]
        del saved
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            (args.output / "config.json").write_text(json.dumps(record, indent=2))
        print(f"DINOv2 Small/16: {world} GPU(s), global batch {args.batch_size}, "
              f"{len(loader):,} steps/epoch, {steps:,} steps", flush=True)
    metrics_path = args.output / "metrics.jsonl"
    timed_images = torch.zeros((), dtype=torch.long, device=device)
    for epoch in range(start_epoch, args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        for index, batch in enumerate(loader):
            if args.benchmark_steps and index == args.benchmark_warmup:
                if world > 1:
                    dist.barrier()
                torch.cuda.synchronize()
                benchmark_start = time.perf_counter()
            step = epoch * len(loader) + index
            progress = step / max(steps - 1, 1)
            ramp = min(1.0, (step + 1) / max(warmup, 1))
            cosine = (1 + math.cos(math.pi * max(0, step - warmup) / max(1, steps - warmup - 1))) / 2
            optimizer.param_groups[0]["lr"] = (1e-6 + (args.lr - 1e-6) * cosine) * ramp
            optimizer.param_groups[0]["weight_decay"] = 0.04 + 0.36 * (1 - cosine)
            momentum = 1 - 0.008 * (1 + math.cos(math.pi * progress)) / 2
            temperature = 0.04 + 0.03 * ramp
            global_views = batch["collated_global_crops"].to(device, non_blocking=True)
            local_views = batch["collated_local_crops"].to(device, non_blocking=True)
            mask = batch["collated_masks"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    target = tokens(teacher["backbone"], global_views)
                    cls_logits = teacher["dino"](target[:, 0]).float()
                    patch_logits = teacher["ibot"](target[:, 1:][mask]).float()
                    cls_prob = dino.softmax_center_teacher(cls_logits, temperature)
                    patch_prob = ibot.softmax_center_teacher(patch_logits.unsqueeze(0), temperature).squeeze(0)
                    dino.update_center(cls_logits)
                    ibot.update_center(patch_logits.unsqueeze(0))
                student_global, student_local, student_patch, features = parallel_student(global_views, local_views, mask)
                cls_loss = cross_view_dino_loss(dino, student_global, student_local, cls_prob, args.local_crops)
                patch_loss = ibot.forward_masked(student_patch, patch_prob, mask)
                diversity = koleo(torch.cat(all_gather(features)) if world > 1 else features)
                loss = cls_loss + patch_loss + 0.1 * diversity
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite SSL loss at step {step + 1}")
            loss.backward()
            if epoch == 0 and args.warmup:
                for name in ("dino", "ibot"):
                    for parameter in student[name].last_layer.parameters():
                        parameter.grad = None
            nn.utils.clip_grad_norm_(student.parameters(), 3.0, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(teacher.parameters()), list(student.parameters()), 1 - momentum)
            if rank == 0 and index % 10 == 0 and not args.benchmark_steps:
                entry = dict(epoch=epoch+1, step=step+1, loss=loss.item(), dino=cls_loss.item(),
                             ibot=patch_loss.item(), koleo=diversity.item())
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
                print(entry, flush=True)
            if args.smoke:
                dino.apply_center_update()
                ibot.apply_center_update()
                if rank == 0:
                    print("Smoke step complete.", flush=True)
                return
            if args.benchmark_steps and index >= args.benchmark_warmup:
                timed_images += global_views.shape[0] // 2
                if index + 1 == args.benchmark_warmup + args.benchmark_steps:
                    dino.apply_center_update()
                    ibot.apply_center_update()
                    if world > 1:
                        dist.all_reduce(timed_images)
                    torch.cuda.synchronize()
                    if rank == 0:
                        elapsed = time.perf_counter() - benchmark_start
                        rate = timed_images.item() / elapsed
                        result = dict(gpus=world, images_per_second=rate,
                                      gpu_seconds_per_1000_images=1000 * world / rate)
                        (args.output / "benchmark.json").write_text(json.dumps(result, indent=2))
                        print(result, flush=True)
                    return
        if args.benchmark_steps:
            raise ValueError("Benchmark must fit within one epoch")
        # Complete the pending distributed center reductions before saving.
        dino.apply_center_update()
        ibot.apply_center_update()
        if rank != 0:
            continue
        destination = args.output / f"teacher_epoch_{epoch + 1:03d}.pth"
        temporary = destination.with_suffix(".part")
        if not args.teacher_every or (epoch + 1) % args.teacher_every == 0 or epoch + 1 in (20, 50, 150, args.epochs):
            torch.save({"teacher": teacher["backbone"].state_dict(), "epoch": epoch + 1,
                        "rfdetr_encoder_prefix": PREFIX}, temporary)
            temporary.replace(destination)
            (args.output / "teacher_path.txt").write_text(f"{destination.resolve()}\n")
            print(f"Saved {destination}", flush=True)
        if args.save_state:
            temporary = args.output / "last.part"
            torch.save(dict(student=student.state_dict(), teacher=teacher.state_dict(), optimizer=optimizer.state_dict(),
                            dino=dino.state_dict(), ibot=ibot.state_dict(), epoch=epoch+1,
                            config={**record, "resume":False}), temporary)
            temporary.replace(args.output / "last.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "images", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--local-crops", type=int, choices=(0, 8), default=8,
                        help="0 keeps two global crops and iBOT, without local-to-global DINO terms")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true", help="Run one training update and exit")
    parser.add_argument("--benchmark-steps", type=int, default=0)
    parser.add_argument("--benchmark-warmup", type=int, default=30)
    parser.add_argument("--teacher-every", type=int, default=0)
    parser.add_argument("--save-state", action="store_true")
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    if not (options.batch_size >= 2 and options.epochs >= 1 and 0 <= options.warmup <= options.epochs):
        parser.error("Require batch size >= 2, epochs >= 1, and warmup in [0, epochs].")
    if options.resume and (not options.save_state or options.benchmark_steps or options.smoke):
        parser.error("--resume requires --save-state and a full training run.")
    if min(options.benchmark_steps, options.benchmark_warmup, options.teacher_every) < 0:
        parser.error("Benchmark and checkpoint intervals must be nonnegative.")
    if options.smoke and options.benchmark_steps:
        parser.error("Choose either --smoke or --benchmark-steps.")
    try:
        train(options)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
