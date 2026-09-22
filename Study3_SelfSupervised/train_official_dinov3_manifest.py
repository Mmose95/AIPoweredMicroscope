"""Run the unmodified official DINOv3 trainer on a manifest-backed dataset.

Launch this file with ``torchrun``. It registers the project-local dataset in
memory and then hands control to the official ``dinov3.train.train`` entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

from dinov3.data import loaders

from dinov3_manifest_dataset import MicroscopyManifest


_official_parse_dataset_str = loaders._parse_dataset_str
_official_init_process_group = torch.distributed.init_process_group


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_public_backbone_initializer(weights_path: Path, output_dir: Path | None) -> None:
    """Strictly load public DINOv3 backbones after the official head init."""
    weights_path = weights_path.expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Approved public DINOv3 weights not found: {weights_path}")
    raw_state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(raw_state, dict) or "patch_embed.proj.weight" not in raw_state:
        raise ValueError("Expected a consolidated DINOv3 backbone state dictionary")
    state = {str(key).removeprefix("module."): value for key, value in raw_state.items()}

    from dinov3.train.ssl_meta_arch import SSLMetaArch
    import dinov3.distributed as distributed
    from torch.distributed.device_mesh import DeviceMesh

    original_init_weights = SSLMetaArch.init_weights

    def distributed_state() -> dict:
        """Match the official consolidated-checkpoint -> FSDP loading path."""
        group = distributed.get_process_subgroup()
        mesh = DeviceMesh.from_group(group, "cuda")
        return {
            key: (
                torch.distributed.tensor.distribute_tensor(tensor, mesh, src_data_rank=None)
                if not any(marker in key for marker in ("rope_embed.periods", "qkv.bias_mask"))
                else tensor
            )
            for key, tensor in state.items()
        }

    def init_weights_with_public_backbone(self):
        # This initializes fresh matching DINO/iBOT heads and copies them to the
        # EMA teacher. Only then are both backbones strictly replaced.
        original_init_weights(self)
        for label, backbone in (("student", self.student.backbone), ("teacher", self.model_ema.backbone)):
            result = backbone.load_state_dict(distributed_state(), strict=True)
            if result.missing_keys or result.unexpected_keys:
                raise RuntimeError(
                    f"Public DINOv3 {label} backbone load was not strict: "
                    f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
                )
        if torch.distributed.get_rank() == 0 and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "public_backbone_initialization.json").write_text(
                json.dumps(
                    {
                        "initialization": "public_dinov3_then_domain_ssl",
                        "weights": str(weights_path),
                        "weights_sha256": _sha256(weights_path),
                        "student_backbone": "strictly_loaded",
                        "teacher_backbone": "strictly_loaded",
                        "ssl_heads": "fresh_random_initialization",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    SSLMetaArch.init_weights = init_weights_with_public_backbone


def _init_process_group_with_explicit_cuda_device(*args, **kwargs):
    """Avoid an NCCL barrier hang on UCloud MIG by binding rank to its GPU."""
    backend = kwargs.get("backend", args[0] if args else None)
    if backend == "nccl" and kwargs.get("device_id") is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        kwargs["device_id"] = torch.device("cuda", local_rank)
    return _official_init_process_group(*args, **kwargs)


def _parse_dataset_str_with_manifest(dataset_str: str):
    prefix = "MicroscopyManifest:"
    if not dataset_str.startswith(prefix):
        return _official_parse_dataset_str(dataset_str)

    kwargs = {}
    for token in dataset_str[len(prefix) :].split(":"):
        if "=" not in token:
            raise ValueError(f"Invalid MicroscopyManifest option: {token!r}")
        key, value = token.split("=", 1)
        if key not in {"manifest", "root"}:
            raise ValueError(f"Unsupported MicroscopyManifest option: {key!r}")
        kwargs[key] = value
    missing = {"manifest", "root"} - kwargs.keys()
    if missing:
        raise ValueError(f"Missing MicroscopyManifest options: {sorted(missing)}")
    return MicroscopyManifest, kwargs


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--public-backbone-weights", type=Path)
    known, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if known.public_backbone_weights is not None:
        output_dir = None
        if "--output-dir" in remaining:
            output_dir = Path(remaining[remaining.index("--output-dir") + 1])
        _install_public_backbone_initializer(known.public_backbone_weights, output_dir)
    loaders._parse_dataset_str = _parse_dataset_str_with_manifest
    torch.distributed.init_process_group = _init_process_group_with_explicit_cuda_device
    # Import after registration so train.py's make_dataset uses the patched parser.
    from dinov3.train.train import main as official_main

    official_main()


if __name__ == "__main__":
    main()
