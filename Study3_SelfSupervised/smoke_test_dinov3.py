"""Local smoke test for a randomly initialized official DINOv3 backbone."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DINOV3_REPO = SCRIPT_DIR.parents[1] / "dinov3"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs" / "smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local forward/backward smoke test with official DINOv3."
    )
    parser.add_argument(
        "--dinov3-repo",
        type=Path,
        default=DEFAULT_DINOV3_REPO,
        help=f"Official DINOv3 checkout (default: {DEFAULT_DINOV3_REPO}).",
    )
    parser.add_argument("--model", default="dinov3_vits16")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for timestamped smoke-test JSON records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.dinov3_repo.resolve()
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(
            f"DINOv3 hubconf.py not found under {repo}. "
            "If the repository is elsewhere, set --dinov3-repo in the PyCharm run configuration."
        )

    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable to PyTorch")

    # pretrained=False is essential: this test must not download or silently
    # load external weights for the scratch/own-data comparison architecture.
    model = torch.hub.load(
        str(repo), args.model, source="local", pretrained=False, trust_repo=True
    ).to(device)
    model.train()
    images = torch.randn(
        args.batch_size, 3, args.image_size, args.image_size, device=device
    )
    output = model(images)
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    if isinstance(tensor, dict):
        tensor = next(value for value in tensor.values() if torch.is_tensor(value))
    loss = tensor.float().square().mean()
    loss.backward()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError("Backward pass did not produce finite gradients")

    created_utc = datetime.now(timezone.utc)
    result = {
        "status": "passed",
        "created_utc": created_utc.isoformat(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "device": device,
        "model": args.model,
        "pretrained": False,
        "input_shape": list(images.shape),
        "output_shape": list(tensor.shape),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "parameters_with_gradients": len(gradients),
        "loss": float(loss.detach().cpu()),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = created_utc.strftime("%Y%m%dT%H%M%SZ")
    result_path = output_dir / f"dinov3_smoke_{timestamp}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"Saved result: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
