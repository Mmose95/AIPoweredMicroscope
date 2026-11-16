"""Prepare RF-DETR Large weights with a controlled DINOv2 encoder initialization."""

from pathlib import Path

from train_dinov2_small_ssl import PREFIX, make_encoder, stock_encoder_state


def write_initialization(stock: Path, output: Path, teacher: Path | None):
    import torch

    payload = torch.load(stock, map_location="cpu", weights_only=False)
    encoder = make_encoder()
    encoder.load_state_dict(stock_encoder_state(payload, encoder), strict=True)
    if teacher is None:
        encoder = make_encoder()
    else:
        exported = torch.load(teacher, map_location="cpu", weights_only=True)
        if exported["rfdetr_encoder_prefix"] != PREFIX:
            raise ValueError(f"Unexpected RF-DETR encoder prefix in {teacher}")
        encoder.load_state_dict(exported["teacher"], strict=True)
    state = payload["model"]
    for name, value in encoder.state_dict().items():
        state[PREFIX + name] = value
    temporary = output.with_suffix(".part")
    torch.save({"model": state, "args": payload.get("args", {})}, temporary)
    temporary.replace(output)
