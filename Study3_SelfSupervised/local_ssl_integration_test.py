"""Tiny native-Windows SSL integration test using an official DINOv3 backbone.

This verifies data flow, teacher/student optimization, checkpoint saving, and
checkpoint reloading. It is not the final DINOv3 research training recipe.

Create randomly initialized DINOv3 student and teacher models.
Perform three teacher–student SSL optimization steps on synthetic images.
Update the teacher using exponential moving averages.
Save the teacher-backbone checkpoint.
Load that checkpoint into a fresh DINOv3 model using strict matching.
Save the result and checkpoint under [runs/ssl_integration](C:/Users/SH37YE/Desktop/PhD_Code_github/AIPoweredMicroscope/Study3_SelfSupervised/runs/ssl_integration).

"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DINOV3_REPO = SCRIPT_DIR.parents[1] / "dinov3"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs" / "ssl_integration"


def latest_local_subset() -> Path | None:
    candidates = sorted((SCRIPT_DIR / "manifests").glob("local_ssl_subset_n*.csv"))
    return candidates[-1] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINOV3_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="dinov3_vits16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--teacher-momentum", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--image-manifest",
        type=Path,
        default=latest_local_subset(),
        help="CSV created by build_local_ssl_subset.py; synthetic images are used if omitted.",
    )
    return parser.parse_args()


def augment(images, torch):
    """Create a lightweight random view while staying entirely on the GPU."""
    view = images.clone()
    flip_mask = torch.rand(view.shape[0], device=view.device) < 0.5
    view[flip_mask] = torch.flip(view[flip_mask], dims=(-1,))
    brightness = 0.8 + 0.4 * torch.rand(view.shape[0], 1, 1, 1, device=view.device)
    contrast = 0.8 + 0.4 * torch.rand(view.shape[0], 1, 1, 1, device=view.device)
    channel_mean = view.mean(dim=(-2, -1), keepdim=True)
    return ((view - channel_mean) * contrast + channel_mean) * brightness


def normalized_loss(student_output, teacher_output, functional):
    student_output = functional.normalize(student_output, dim=-1)
    teacher_output = functional.normalize(teacher_output.detach(), dim=-1)
    return 2.0 - 2.0 * (student_output * teacher_output).sum(dim=-1).mean()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("--steps and --batch-size must be positive")
    if not 0.0 <= args.teacher_momentum < 1.0:
        raise ValueError("--teacher-momentum must be in [0, 1)")

    repo = args.dinov3_repo.resolve()
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv3 hubconf.py not found under {repo}")

    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from torchvision.transforms import v2

    torch.manual_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    student = torch.hub.load(
        str(repo), args.model, source="local", pretrained=False, trust_repo=True
    ).to(device)
    teacher = copy.deepcopy(student).to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    feature_dim = 384 if args.model in {"dinov3_vits16", "dinov3_vits16plus"} else None
    if feature_dim is None:
        with torch.no_grad():
            probe = student(torch.zeros(1, 3, args.image_size, args.image_size, device=device))
        feature_dim = int(probe.shape[-1])
    student_head = torch.nn.Sequential(
        torch.nn.Linear(feature_dim, 512), torch.nn.GELU(), torch.nn.Linear(512, 128)
    ).to(device)
    teacher_head = copy.deepcopy(student_head).to(device).eval()
    for parameter in teacher_head.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        [*student.parameters(), *student_head.parameters()], lr=args.learning_rate
    )
    image_paths = []
    if args.image_manifest is not None:
        manifest_path = args.image_manifest.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Image manifest not found: {manifest_path}")
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            image_paths = [Path(row["local_absolute_path"]) for row in csv.DictReader(handle)]
        missing = [str(path) for path in image_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing manifest images; first examples: {missing[:5]}")
    else:
        manifest_path = None
    image_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((args.image_size, args.image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    generator = torch.Generator().manual_seed(args.seed)
    image_order = (
        torch.randperm(len(image_paths), generator=generator).tolist() if image_paths else []
    )
    losses = []
    student.train()
    for step in range(args.steps):
        if image_paths:
            start = (step * args.batch_size) % len(image_order)
            indices = [image_order[(start + offset) % len(image_order)] for offset in range(args.batch_size)]
            loaded = []
            for index in indices:
                with Image.open(image_paths[index]) as image:
                    loaded.append(image_transform(image.convert("RGB")))
            images = torch.stack(loaded).to(device)
        else:
            images = torch.randn(
                args.batch_size, 3, args.image_size, args.image_size, device=device
            )
        view_a, view_b = augment(images, torch), augment(images, torch)
        with torch.no_grad():
            teacher_a = teacher_head(teacher(view_a))
            teacher_b = teacher_head(teacher(view_b))
        student_a = student_head(student(view_a))
        student_b = student_head(student(view_b))
        loss = 0.5 * (
            normalized_loss(student_a, teacher_b, functional)
            + normalized_loss(student_b, teacher_a, functional)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            momentum = args.teacher_momentum
            for teacher_parameter, student_parameter in zip(
                teacher.parameters(), student.parameters()
            ):
                teacher_parameter.mul_(momentum).add_(student_parameter, alpha=1 - momentum)
            for teacher_parameter, student_parameter in zip(
                teacher_head.parameters(), student_head.parameters()
            ):
                teacher_parameter.mul_(momentum).add_(student_parameter, alpha=1 - momentum)
        losses.append(float(loss.detach().cpu()))
        print(f"step={step + 1}/{args.steps} loss={losses[-1]:.6f}")

    created_utc = datetime.now(timezone.utc)
    run_dir = args.output_dir.resolve() / created_utc.strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "integration_teacher_checkpoint.pth"
    torch.save(
        {
            "purpose": "integration_test_only_not_research_training",
            "method": "two_view_ema_teacher_smoke_test",
            "model": args.model,
            "pretrained": False,
            "teacher_backbone": teacher.state_dict(),
            "teacher_projection_head": teacher_head.state_dict(),
            "steps": args.steps,
            "seed": args.seed,
            "image_source": "manifest" if image_paths else "synthetic",
            "image_manifest": str(manifest_path) if manifest_path else None,
        },
        checkpoint_path,
    )

    reloaded = torch.hub.load(
        str(repo), args.model, source="local", pretrained=False, trust_repo=True
    ).cpu()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_result = reloaded.load_state_dict(payload["teacher_backbone"], strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint reload failed: {load_result}")

    result = {
        "status": "passed",
        "purpose": "integration_test_only_not_research_training",
        "created_utc": created_utc.isoformat(),
        "model": args.model,
        "pretrained": False,
        "device": device,
        "torch_version": torch.__version__,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "image_source": "manifest" if image_paths else "synthetic",
        "image_manifest": str(manifest_path) if manifest_path else None,
        "available_manifest_images": len(image_paths),
        "losses": losses,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "strict_reload": True,
    }
    result_path = run_dir / "integration_result.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved result: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
