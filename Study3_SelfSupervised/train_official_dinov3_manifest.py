"""Run the unmodified official DINOv3 trainer on a manifest-backed dataset.

Launch this file with ``torchrun``. It registers the project-local dataset in
memory and then hands control to the official ``dinov3.train.train`` entrypoint.
"""

from __future__ import annotations

from dinov3.data import loaders

from dinov3_manifest_dataset import MicroscopyManifest


_official_parse_dataset_str = loaders._parse_dataset_str


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
    # Import after registration so train.py's make_dataset uses the patched parser.
    from dinov3.train.train import main as official_main

    official_main()


if __name__ == "__main__":
    main()

