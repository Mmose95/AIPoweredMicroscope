"""Single-GPU DINOv2 SSL for the RF-DETR Large Small/16 encoder."""
import argparse
import csv
from functools import partial
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dinov2.data.augmentations import DataAugmentationDINO
from dinov2.data.collate import collate_data_and_cast
from dinov2.data.masking import MaskingGenerator
from dinov2.layers.dino_head import DINOHead
from dinov2.loss import DINOLoss, KoLeoLoss, iBOTPatchLoss
from rfdetr.models.backbone.dinov2 import DinoV2

PREFIX = "backbone.0.encoder.encoder."


class ManifestImages(Dataset):
    def __init__(self, manifest, root):
        with manifest.open(encoding="utf-8", newline="") as handle:
            self.paths = [root / row["relative_path"] for row in csv.DictReader(handle)]
        if len(self.paths) < 2:
            raise ValueError("SSL needs at least two training images")
        self.transform = DataAugmentationDINO((0.32, 1.0), (0.05, 0.32), 8, 224, 96)

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


def tokens(encoder, images, masks=None):
    embeddings = encoder.embeddings(images, bool_masked_pos=masks)
    encoded = encoder.encoder(embeddings, return_dict=False)[0]
    return encoder.layernorm(encoded)


def ssl_network(encoder):
    return nn.ModuleDict({
        "backbone": encoder, "dino": DINOHead(384, 65536), "ibot": DINOHead(384, 65536),
    })


def train(args):
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda")
    dataset = ManifestImages(args.manifest, args.images)
    collate = partial(collate_data_and_cast, mask_ratio_tuple=(0.1, 0.5),
                      mask_probability=0.5, dtype=torch.float32, n_tokens=196,
                      mask_generator=MaskingGenerator((14, 14), max_num_patches=98))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=collate, pin_memory=True, persistent_workers=args.workers > 0)
    encoder = make_encoder()
    if args.initial_checkpoint:
        source = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        encoder.load_state_dict({name.removeprefix(PREFIX): value for name, value in source["model"].items()
                                 if name.startswith(PREFIX)}, strict=True)
    student = ssl_network(encoder)
    teacher = ssl_network(make_encoder()).requires_grad_(False).eval()
    teacher.load_state_dict(student.state_dict())
    student.to(device)
    teacher.to(device)
    dino = DINOLoss(65536).to(device)
    ibot = iBOTPatchLoss(65536).to(device)
    koleo = KoLeoLoss()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    steps = len(loader) * args.epochs
    warmup = len(loader) * args.warmup
    args.output.mkdir(parents=True, exist_ok=True)
    record = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output / "config.json").write_text(json.dumps(record | {"images_in_manifest":len(dataset),
        "steps_per_epoch":len(loader), "torch":torch.__version__, "rfdetr":version("rfdetr")}, indent=2))
    print(f"DINOv2 Small/16: {len(dataset):,} images, {len(loader):,} steps/epoch, {steps:,} steps", flush=True)
    metrics_path = args.output / "metrics.jsonl"
    for epoch in range(args.epochs):
        for index, batch in enumerate(loader):
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
                global_tokens = tokens(student["backbone"], global_views, mask)
                local_tokens = tokens(student["backbone"], local_views)
                s0, s1 = student["dino"](global_tokens[:, 0]).float().chunk(2)
                t0, t1 = cls_prob.chunk(2)
                local = student["dino"](local_tokens[:, 0]).float().chunk(8)
                cls_loss = (dino([s0], [t1]) + dino([s1], [t0]) + dino(local, [t0, t1])) / 18
                patch_loss = ibot.forward_masked(student["ibot"](global_tokens[:, 1:][mask]).float(),
                                                 patch_prob, mask)
                diversity = koleo(global_tokens[:, 0].float())
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
            if index % 10 == 0:
                entry = dict(epoch=epoch+1, step=step+1, loss=loss.item(), dino=cls_loss.item(),
                             ibot=patch_loss.item(), koleo=diversity.item())
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry) + "\n")
                print(entry, flush=True)
            if args.smoke:
                print("Smoke step complete.", flush=True)
                return
        destination = args.output / f"teacher_epoch_{epoch + 1:03d}.pth"
        temporary = destination.with_suffix(".part")
        torch.save({"teacher": teacher["backbone"].state_dict(), "epoch": epoch + 1,
                    "rfdetr_encoder_prefix": PREFIX}, temporary)
        temporary.replace(destination)
        (args.output / "teacher_path.txt").write_text(f"{destination.resolve()}\n")
        print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "images", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--smoke", action="store_true", help="Run one training update and exit")
    options = parser.parse_args()
    if not (options.batch_size >= 2 and options.epochs >= 1 and 0 <= options.warmup <= options.epochs):
        parser.error("Require batch size >= 2, epochs >= 1, and warmup in [0, epochs].")
    if (options.output / "config.json").exists():
        parser.error("Choose a fresh output directory.")
    train(options)
