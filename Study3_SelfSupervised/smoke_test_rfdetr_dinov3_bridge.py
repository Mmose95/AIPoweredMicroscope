"""Smoke-test the native DINOv3 feature encoder and optional RF-DETR install."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rfdetr_dinov3_bridge import (
    DinoV3FeatureEncoder,
    install_dinov3_encoder,
    patch_rfdetr_training_rebuild,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DINOV3_REPO = SCRIPT_DIR.parents[1] / "dinov3"
DEFAULT_LOCAL_CHECKPOINT = Path(
    r"E:\PHD\Results\SSL_QA40X\dinov3_vits16_full_seed0"
    r"\eval\training_81623\teacher_checkpoint.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINOV3_REPO)
    parser.add_argument(
        "--initialization",
        choices=("scratch", "public_ssl", "own_data_ssl"),
        default="own_data_ssl",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--public-ssl-weights", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--test-rfdetr",
        action="store_true",
        help="Also instantiate RF-DETR Small and replace its complete encoder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.initialization == "own_data_ssl" and args.checkpoint is None:
        args.checkpoint = DEFAULT_LOCAL_CHECKPOINT
    if args.initialization in ("scratch", "public_ssl") and args.checkpoint is not None:
        raise ValueError("--checkpoint is forbidden for scratch or public_ssl initialization")
    if args.initialization == "own_data_ssl" and args.checkpoint is None:
        raise ValueError("--checkpoint is required for own_data_ssl initialization")
    if args.initialization == "public_ssl" and args.public_ssl_weights is None:
        raise ValueError("--public-ssl-weights is required for public_ssl initialization")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    encoder = DinoV3FeatureEncoder(
        dinov3_repo=args.dinov3_repo,
        initialization=args.initialization,
        checkpoint=args.checkpoint,
        public_weights=args.public_ssl_weights,
    ).to(device)
    images = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    features = encoder(images)
    loss = sum(feature.float().square().mean() for feature in features)
    loss.backward()
    result = {
        "status": "passed",
        "device": device,
        "provenance": encoder.provenance_dict(),
        "input_shape": list(images.shape),
        "feature_shapes": [list(feature.shape) for feature in features],
        "finite_features": all(bool(torch.isfinite(feature).all()) for feature in features),
        "parameters_with_gradients": sum(p.grad is not None for p in encoder.parameters()),
        "rfdetr_installed": False,
    }

    if args.test_rfdetr:
        from rfdetr import RFDETRSmall

        # RF-DETR does not load a detector checkpoint when pretrain_weights=None.
        # Its temporary encoder is replaced in full immediately below.
        rf_model = RFDETRSmall(pretrain_weights=None, resolution=args.image_size, patch_size=16)
        installed = install_dinov3_encoder(
            rf_model,
            dinov3_repo=args.dinov3_repo,
            initialization=args.initialization,
            checkpoint=args.checkpoint,
            public_weights=args.public_ssl_weights,
        ).to(device)
        installed_features = installed(images.detach())
        result["rfdetr_installed"] = True
        result["rfdetr_feature_shapes"] = [list(feature.shape) for feature in installed_features]
        result["same_encoder_object"] = rf_model.model.model.backbone[0].encoder is installed

        # RF-DETR 1.9 creates a fresh detector inside train(). Verify that the
        # scoped training hook replaces that newly built encoder as well.
        from rfdetr.training import RFDETRModelModule

        train_config = rf_model.get_train_config(dataset_dir=".", output_dir=".")
        with patch_rfdetr_training_rebuild(
            dinov3_repo=args.dinov3_repo,
            initialization=args.initialization,
            checkpoint=args.checkpoint,
            public_weights=args.public_ssl_weights,
        ):
            training_module = RFDETRModelModule(rf_model.model_config, train_config)
        rebuilt_encoder = training_module.model.backbone[0].encoder.to(device)
        rebuilt_features = rebuilt_encoder(images.detach())
        result["training_rebuild_intercepted"] = isinstance(
            rebuilt_encoder,
            DinoV3FeatureEncoder,
        )
        result["training_rebuild_initialization"] = rebuilt_encoder.provenance.initialization
        result["training_rebuild_feature_shapes"] = [
            list(feature.shape) for feature in rebuilt_features
        ]

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
