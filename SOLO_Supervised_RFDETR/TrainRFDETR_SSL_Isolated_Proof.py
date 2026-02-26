from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall


TARGET_EMBED_PREFIX = "backbone.0.encoder.encoder.embeddings."
TARGET_BLOCK_PREFIX = "backbone.0.encoder.encoder.encoder.layer."
TARGET_NORM_PREFIX = "backbone.0.encoder.encoder.layernorm."
KNOWN_STRIP_PREFIXES = (
    "module.",
    "state_dict.",
    "model.",
    "teacher.",
    "student.",
    "backbone.",
    "encoder.",
)
BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(\d+)\.(.+)$")


def _strip_known_prefixes(key: str) -> str:
    changed = True
    out = key
    while changed:
        changed = False
        for pref in KNOWN_STRIP_PREFIXES:
            if out.startswith(pref):
                out = out[len(pref):]
                changed = True
    return out


def _extract_state_dict(payload: dict) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for k in ("state_dict", "model", "teacher", "student"):
            v = payload.get(k)
            if isinstance(v, dict) and v:
                first_val = next(iter(v.values()))
                if torch.is_tensor(first_val):
                    return v
    if isinstance(payload, dict) and payload:
        first_val = next(iter(payload.values()))
        if torch.is_tensor(first_val):
            return payload
    raise ValueError("Could not find a tensor state_dict in SSL checkpoint.")


def _try_add(
    mapped: dict[str, torch.Tensor],
    target_sd: dict[str, torch.Tensor],
    dst_key: str,
    value: torch.Tensor,
    stats: dict[str, int],
    bucket: str,
) -> bool:
    if dst_key not in target_sd:
        return False
    if tuple(value.shape) != tuple(target_sd[dst_key].shape):
        stats["shape_mismatch"] += 1
        return False
    mapped[dst_key] = value
    stats[bucket] += 1
    return True


def _map_ssl_to_rfdetr_backbone(
    ssl_sd: dict[str, torch.Tensor], target_sd: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    mapped: dict[str, torch.Tensor] = {}
    stats = {
        "tensor_keys": 0,
        "direct_matches": 0,
        "converted_matches": 0,
        "shape_mismatch": 0,
        "unmapped": 0,
    }

    simple_global_map = {
        "cls_token": f"{TARGET_EMBED_PREFIX}cls_token",
        "mask_token": f"{TARGET_EMBED_PREFIX}mask_token",
        "pos_embed": f"{TARGET_EMBED_PREFIX}position_embeddings",
        "patch_embed.proj.weight": f"{TARGET_EMBED_PREFIX}patch_embeddings.projection.weight",
        "patch_embed.proj.bias": f"{TARGET_EMBED_PREFIX}patch_embeddings.projection.bias",
        "norm.weight": f"{TARGET_NORM_PREFIX}weight",
        "norm.bias": f"{TARGET_NORM_PREFIX}bias",
    }

    simple_block_map = {
        "norm1.weight": "norm1.weight",
        "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
        "mlp.fc1.weight": "mlp.fc1.weight",
        "mlp.fc1.bias": "mlp.fc1.bias",
        "mlp.fc2.weight": "mlp.fc2.weight",
        "mlp.fc2.bias": "mlp.fc2.bias",
        "attn.proj.weight": "attention.output.dense.weight",
        "attn.proj.bias": "attention.output.dense.bias",
    }

    for src_key, value in ssl_sd.items():
        if not torch.is_tensor(value):
            continue
        stats["tensor_keys"] += 1
        used = False

        for direct_key in (
            src_key,
            f"backbone.0.{src_key}",
            f"backbone.0.encoder.{src_key}",
            f"backbone.0.encoder.encoder.{src_key}",
        ):
            if _try_add(mapped, target_sd, direct_key, value, stats, "direct_matches"):
                used = True
                break
        if used:
            continue

        norm_key = _strip_known_prefixes(src_key)
        if norm_key != src_key:
            for direct_key in (
                norm_key,
                f"backbone.0.{norm_key}",
                f"backbone.0.encoder.{norm_key}",
                f"backbone.0.encoder.encoder.{norm_key}",
            ):
                if _try_add(mapped, target_sd, direct_key, value, stats, "direct_matches"):
                    used = True
                    break
            if used:
                continue

        if norm_key in simple_global_map:
            if _try_add(mapped, target_sd, simple_global_map[norm_key], value, stats, "converted_matches"):
                continue

        m = BLOCK_RE.match(norm_key)
        if not m:
            stats["unmapped"] += 1
            continue

        _, layer_idx, suffix = m.groups()
        layer_prefix = f"{TARGET_BLOCK_PREFIX}{int(layer_idx)}."

        if suffix == "attn.qkv.weight":
            q, k, v = value.chunk(3, dim=0)
            ok_q = _try_add(mapped, target_sd, f"{layer_prefix}attention.attention.query.weight", q, stats, "converted_matches")
            ok_k = _try_add(mapped, target_sd, f"{layer_prefix}attention.attention.key.weight", k, stats, "converted_matches")
            ok_v = _try_add(mapped, target_sd, f"{layer_prefix}attention.attention.value.weight", v, stats, "converted_matches")
            if ok_q and ok_k and ok_v:
                continue
            stats["unmapped"] += 1
            continue

        if suffix == "attn.qkv.bias":
            q, k, v = value.chunk(3, dim=0)
            ok_q = _try_add(mapped, target_sd, f"{layer_prefix}attention.attention.query.bias", q, stats, "converted_matches")
            ok_k = _try_add(mapped, target_sd, f"{layer_prefix}attention.attention.key.bias", k, stats, "converted_matches")
            ok_v = _try_add(mapped, target_sd, f"{layer_prefix}attention.attention.value.bias", v, stats, "converted_matches")
            if ok_q and ok_k and ok_v:
                continue
            stats["unmapped"] += 1
            continue

        mapped_suffix = simple_block_map.get(suffix)
        if mapped_suffix is None:
            stats["unmapped"] += 1
            continue
        if not _try_add(mapped, target_sd, f"{layer_prefix}{mapped_suffix}", value, stats, "converted_matches"):
            stats["unmapped"] += 1

    return mapped, stats


def load_ssl_backbone_into_rfdetr(
    rf_model, ssl_ckpt: Path, min_loaded_keys: int
) -> dict:
    raw = torch.load(str(ssl_ckpt), map_location="cpu", weights_only=False)
    ssl_sd = _extract_state_dict(raw)
    target_model = rf_model.model.model
    target_sd = target_model.state_dict()
    mapped_sd, stats = _map_ssl_to_rfdetr_backbone(ssl_sd, target_sd)

    if len(mapped_sd) < int(min_loaded_keys):
        raise RuntimeError(
            f"Only {len(mapped_sd)} SSL keys mapped into RF-DETR; min_loaded_keys={min_loaded_keys}."
        )

    sentinel_key = next(iter(mapped_sd.keys()))
    before_norm = float(target_sd[sentinel_key].detach().float().norm().item())
    load_info = target_model.load_state_dict(mapped_sd, strict=False)
    after_norm = float(target_model.state_dict()[sentinel_key].detach().float().norm().item())

    report = {
        "ssl_ckpt": str(ssl_ckpt),
        "mapped_key_count": int(len(mapped_sd)),
        "stats": stats,
        "sentinel_key": sentinel_key,
        "sentinel_norm_before": before_norm,
        "sentinel_norm_after": after_norm,
        "sentinel_changed": bool(abs(after_norm - before_norm) > 1e-12),
        "missing_keys_count": int(len(load_info.missing_keys)),
        "unexpected_keys_count": int(len(load_info.unexpected_keys)),
        "missing_keys_sample": load_info.missing_keys[:25],
        "unexpected_keys_sample": load_info.unexpected_keys[:25],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated proof script: explicitly load SSL checkpoint into RF-DETR backbone, then train."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ssl-ckpt", type=Path, required=True)
    parser.add_argument("--class-name", type=str, default="target")
    parser.add_argument("--model", type=str, default="large", choices=("small", "medium", "large"))
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=7e-4)
    parser.add_argument("--num-queries", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-loaded-keys", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.ssl_ckpt.exists():
        raise FileNotFoundError(f"SSL checkpoint not found: {args.ssl_ckpt}")
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {args.dataset_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_meta = args.output_dir / "run_meta"
    run_meta.mkdir(parents=True, exist_ok=True)

    model_cls = {
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }[args.model]
    rf_model = model_cls(pretrain_weights=None, resolution=int(args.resolution))

    report = load_ssl_backbone_into_rfdetr(
        rf_model=rf_model,
        ssl_ckpt=args.ssl_ckpt,
        min_loaded_keys=int(args.min_loaded_keys),
    )
    print(json.dumps(report, indent=2))
    (run_meta / "ssl_load_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.dry_run:
        print("[DONE] Dry-run only. Skipping training.")
        return

    rf_model.train(
        dataset_dir=str(args.dataset_dir),
        output_dir=str(args.output_dir),
        class_names=[args.class_name],
        resolution=int(args.resolution),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        grad_accum_steps=int(args.grad_accum_steps),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_queries=int(args.num_queries),
        num_workers=int(args.num_workers),
        run_test=True,
        early_stopping=False,
        amp=True,
    )


if __name__ == "__main__":
    main()

