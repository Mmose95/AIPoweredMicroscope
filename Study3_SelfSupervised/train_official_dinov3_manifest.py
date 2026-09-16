"""Run the unmodified official DINOv3 trainer on a manifest-backed dataset.

Launch this file with ``torchrun``. It registers the project-local dataset in
memory and then hands control to the official ``dinov3.train.train`` entrypoint.
"""

from __future__ import annotations

import os

import torch

from dinov3.data import loaders

from dinov3_manifest_dataset import MicroscopyManifest


_official_parse_dataset_str = loaders._parse_dataset_str
_official_init_process_group = torch.distributed.init_process_group


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
    loaders._parse_dataset_str = _parse_dataset_str_with_manifest
    torch.distributed.init_process_group = _init_process_group_with_explicit_cuda_device
    # Import after registration so train.py's make_dataset uses the patched parser.
    from dinov3.train.train import main as official_main

    official_main()


if __name__ == "__main__":
    main()
