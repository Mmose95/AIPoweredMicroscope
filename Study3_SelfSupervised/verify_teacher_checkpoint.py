"""Strictly load an exported official DINOv3 EMA-teacher backbone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DINOV3_REPO = SCRIPT_DIR.parents[1] / "dinov3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINOV3_REPO)
    parser.add_argument("--model", default="dinov3_vits16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    checkpoint = args.checkpoint.expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "teacher" not in payload:
        raise KeyError(f"Checkpoint has no 'teacher' state: {checkpoint}")
    state = {}
    for key, value in payload["teacher"].items():
        key = key.removeprefix("module.")
        # The official eval export contains the EMA backbone and its SSL heads.
        # A downstream detector needs only the backbone tensors.
        if not key.startswith("backbone."):
            continue
        state[key.removeprefix("backbone.")] = value
    if not state:
        raise RuntimeError("No 'backbone.' tensors found in teacher checkpoint")
    model = torch.hub.load(
        str(args.dinov3_repo.resolve()),
        args.model,
        source="local",
        pretrained=False,
        trust_repo=True,
    )
    load_result = model.load_state_dict(state, strict=True)
    result = {
        "status": "passed",
        "checkpoint": str(checkpoint),
        "model": args.model,
        "strict_load": True,
        "missing_keys": load_result.missing_keys,
        "unexpected_keys": load_result.unexpected_keys,
        "parameter_tensors": len(state),
        "ssl_heads_excluded": True,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
