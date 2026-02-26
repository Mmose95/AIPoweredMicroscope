from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
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
BLOCK_RE_SINGLE = re.compile(r"^blocks\.(\d+)\.(.+)$")
VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


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


def _parse_block_key(norm_key: str) -> tuple[int, str] | None:
    m = BLOCK_RE.match(norm_key)
    if m:
        _, layer_idx, suffix = m.groups()
        return int(layer_idx), suffix
    m = BLOCK_RE_SINGLE.match(norm_key)
    if m:
        layer_idx, suffix = m.groups()
        return int(layer_idx), suffix
    return None


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

        parsed = _parse_block_key(norm_key)
        if parsed is None:
            stats["unmapped"] += 1
            continue

        layer_idx, suffix = parsed
        layer_prefix = f"{TARGET_BLOCK_PREFIX}{layer_idx}."

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
) -> tuple[dict, dict[str, torch.Tensor]]:
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
    return report, mapped_sd


def _pick_monitor_key(mapped_sd: dict[str, torch.Tensor]) -> str:
    preferred_suffixes = (
        "embeddings.patch_embeddings.projection.weight",
        "embeddings.cls_token",
        "encoder.layer.0.attention.attention.query.weight",
    )
    keys = list(mapped_sd.keys())
    for suffix in preferred_suffixes:
        for k in keys:
            if k.endswith(suffix):
                return k
    return keys[0]


def _state_tensor_for_key(model_sd: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key in model_sd:
        return model_sd[key]
    module_key = f"module.{key}"
    if module_key in model_sd:
        return model_sd[module_key]
    raise KeyError(f"Could not find monitor key in state_dict: {key}")


def _install_train_start_probe(rf_model, monitor_key: str, expected_tensor: torch.Tensor, strict: bool = True) -> dict:
    probe = {
        "monitor_key": monitor_key,
        "first_batch_seen": False,
        "first_batch_step": None,
        "first_batch_max_abs_diff": None,
        "train_end_seen": False,
        "post_train_max_abs_diff": None,
    }

    expected = expected_tensor.detach().cpu()

    def _on_train_batch_start(ctx: dict):
        if probe["first_batch_seen"]:
            return
        state = ctx["model"].state_dict()
        cur = _state_tensor_for_key(state, monitor_key).detach().cpu()
        max_abs = float((cur - expected).abs().max().item())
        probe["first_batch_seen"] = True
        probe["first_batch_step"] = int(ctx.get("step", -1))
        probe["first_batch_max_abs_diff"] = max_abs
        print(
            f"[SSL-PROBE] first_batch step={probe['first_batch_step']} "
            f"monitor_key={monitor_key} max_abs_diff={max_abs:.3e}"
        )
        if strict and max_abs > 1e-6:
            raise RuntimeError(
                f"Training did not start from loaded SSL weights for {monitor_key}. "
                f"max_abs_diff={max_abs:.6e}"
            )

    def _on_train_end():
        final_sd = rf_model.model.model.state_dict()
        final = _state_tensor_for_key(final_sd, monitor_key).detach().cpu()
        max_abs = float((final - expected).abs().max().item())
        probe["train_end_seen"] = True
        probe["post_train_max_abs_diff"] = max_abs
        print(f"[SSL-PROBE] train_end monitor_key={monitor_key} max_abs_diff_from_loaded={max_abs:.3e}")

    rf_model.callbacks["on_train_batch_start"].append(_on_train_batch_start)
    rf_model.callbacks["on_train_end"].append(_on_train_end)
    return probe


def _norm_rel(s: str) -> str:
    return str(s).replace("\\", "/").lstrip("./")


def _build_image_index(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_rel: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = defaultdict(list)
    root = root.resolve()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            rel = p.resolve().relative_to(root).as_posix()
            by_rel[rel] = p.resolve()
            by_name[p.name].append(p.resolve())
    return by_rel, by_name


def _resolve_image_path(
    file_name: str,
    dataset_dir: Path,
    split: str,
    fallback_root: Path | None,
    index_rel: dict[str, Path] | None,
    index_name: dict[str, list[Path]] | None,
) -> Path | None:
    fn = str(file_name).strip()
    p = Path(fn)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    split_dir = dataset_dir / split
    candidates.extend(
        [
            split_dir / fn,
            split_dir / "images" / fn,
            dataset_dir / fn,
            dataset_dir / "images" / fn,
        ]
    )
    if fallback_root is not None:
        candidates.extend(
            [
                fallback_root / fn,
                fallback_root / "images" / fn,
            ]
        )
    for c in candidates:
        if c.exists():
            return c.resolve()
    if index_rel is not None:
        rel = _norm_rel(fn)
        if rel in index_rel:
            return index_rel[rel]
    if index_name is not None:
        base = Path(fn).name
        matches = index_name.get(base, [])
        if len(matches) == 1:
            return matches[0]
    return None


def build_resolved_dataset(
    dataset_dir: Path,
    out_dir: Path,
    images_fallback_root: Path | None,
) -> tuple[Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_in": str(dataset_dir),
        "dataset_out": str(out_dir),
        "images_fallback_root": str(images_fallback_root) if images_fallback_root else None,
        "splits": {},
    }

    index_rel = None
    index_name = None
    if images_fallback_root is not None and images_fallback_root.exists():
        index_rel, index_name = _build_image_index(images_fallback_root)

    for split in ("train", "valid", "test"):
        src_json = dataset_dir / split / "_annotations.coco.json"
        if not src_json.exists():
            continue
        js = json.loads(src_json.read_text(encoding="utf-8"))
        out_split = out_dir / split
        out_split.mkdir(parents=True, exist_ok=True)

        misses = []
        updated = 0
        for im in js.get("images", []):
            resolved = _resolve_image_path(
                file_name=str(im.get("file_name", "")),
                dataset_dir=dataset_dir,
                split=split,
                fallback_root=images_fallback_root,
                index_rel=index_rel,
                index_name=index_name,
            )
            if resolved is None:
                misses.append(str(im.get("file_name", "")))
                continue
            im["file_name"] = str(resolved)
            updated += 1

        if misses:
            preview = "\n".join(misses[:8])
            raise FileNotFoundError(
                f"Could not resolve {len(misses)} image paths in split '{split}'. "
                f"Examples:\n{preview}\n"
                f"Try setting --images-fallback-root to the actual image root."
            )

        (out_split / "_annotations.coco.json").write_text(
            json.dumps(js, indent=2),
            encoding="utf-8",
        )
        report["splits"][split] = {
            "images": int(len(js.get("images", []))),
            "annotations": int(len(js.get("annotations", []))),
            "resolved_images": int(updated),
        }

    return out_dir, report


def _default_images_fallback_root() -> Path | None:
    raw = os.getenv("IMAGES_FALLBACK_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser()
    candidates = [
        Path(r"D:\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment_Cleaned"),
        Path.cwd() / "CellScanData" / "Zoom10x - Quality Assessment_Cleaned",
        Path(__file__).resolve().parents[1] / "CellScanData" / "Zoom10x - Quality Assessment_Cleaned",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def _default_dataset_dir() -> Path:
    raw = os.getenv("DATASET_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    candidates = [
        Path(
            r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\SOLO_Supervised_RFDETR\Stat_Dataset\QA-2025v2_SquamousEpithelialCell_OVR_20260217-093944"
        ),
        Path(__file__).resolve().parent / "Stat_Dataset" / "QA-2025v2_SquamousEpithelialCell_OVR_20260217-093944",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0]


def _default_output_dir() -> Path:
    raw = os.getenv("OUTPUT_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(
        r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\SOLO_Supervised_RFDETR\RFDETR_SOLO_OUTPUT\ssl_isolated_epi"
    )


def _default_ssl_ckpt() -> Path:
    raw = os.getenv("SSL_CKPT", "").strip()
    if raw:
        return Path(raw).expanduser()
    candidates = [
        Path(r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\dinov2_base_selfsup_trained.pt"),
        Path(r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\Checkpoints\checkpoint.pth"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated proof script: explicitly load SSL checkpoint into RF-DETR backbone, then train."
    )
    parser.add_argument("--dataset-dir", type=Path, default=_default_dataset_dir())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--ssl-ckpt", type=Path, default=_default_ssl_ckpt())
    parser.add_argument("--images-fallback-root", type=Path, default=_default_images_fallback_root())
    parser.add_argument("--skip-resolve-dataset", action="store_true")
    parser.add_argument("--class-name", type=str, default="Squamous Epithelial Cell")
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
    parser.add_argument("--min-batches", type=int, default=1)
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--non-strict-train-start", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.dataset_dir = args.dataset_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.ssl_ckpt = args.ssl_ckpt.expanduser()
    if args.images_fallback_root is not None:
        args.images_fallback_root = args.images_fallback_root.expanduser()

    if not args.ssl_ckpt.exists():
        raise FileNotFoundError(f"SSL checkpoint not found: {args.ssl_ckpt}")
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {args.dataset_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_meta = args.output_dir / "run_meta"
    run_meta.mkdir(parents=True, exist_ok=True)

    effective_dataset_dir = args.dataset_dir
    if not args.skip_resolve_dataset:
        resolved_dir = args.output_dir / "_resolved_dataset"
        effective_dataset_dir, resolved_report = build_resolved_dataset(
            dataset_dir=args.dataset_dir,
            out_dir=resolved_dir,
            images_fallback_root=args.images_fallback_root,
        )
        (run_meta / "resolved_dataset_report.json").write_text(
            json.dumps(resolved_report, indent=2),
            encoding="utf-8",
        )

    model_cls = {
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
    }[args.model]
    rf_model = model_cls(pretrain_weights=None, resolution=int(args.resolution))

    report, mapped_sd = load_ssl_backbone_into_rfdetr(
        rf_model=rf_model,
        ssl_ckpt=args.ssl_ckpt,
        min_loaded_keys=int(args.min_loaded_keys),
    )
    monitor_key = _pick_monitor_key(mapped_sd)
    loaded_monitor = mapped_sd[monitor_key].detach().cpu()
    current_monitor = rf_model.model.model.state_dict()[monitor_key].detach().cpu()
    pretrain_max_abs_diff = float((current_monitor - loaded_monitor).abs().max().item())
    report["monitor_key"] = monitor_key
    report["pretrain_max_abs_diff"] = pretrain_max_abs_diff
    report["pretrain_exact_match"] = bool(pretrain_max_abs_diff <= 1e-6)
    if pretrain_max_abs_diff > 1e-6:
        raise RuntimeError(
            f"Post-load verification failed for {monitor_key}. max_abs_diff={pretrain_max_abs_diff:.6e}"
        )

    print(json.dumps(report, indent=2))
    (run_meta / "ssl_load_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.dry_run:
        print("[DONE] Dry-run only. Skipping training.")
        return

    probe = _install_train_start_probe(
        rf_model=rf_model,
        monitor_key=monitor_key,
        expected_tensor=loaded_monitor,
        strict=not args.non_strict_train_start,
    )

    try:
        rf_model.train(
            dataset_dir=str(effective_dataset_dir),
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
            min_batches=int(args.min_batches),
            run_test=bool(args.run_test),
            early_stopping=False,
            amp=True,
        )
    except FileNotFoundError as e:
        msg = str(e)
        if "checkpoint_best_regular.pth" in msg or "checkpoint_best_total.pth" in msg:
            probe["train_exception_ignored"] = msg
            print(
                "[WARN] RF-DETR finished epochs but did not create a best checkpoint "
                "(often happens when mAP never exceeds 0.0)."
            )
        else:
            raise

    if not probe["first_batch_seen"]:
        raise RuntimeError("Training probe failed: no training batch callback observed.")
    probe_path = run_meta / "ssl_train_probe_report.json"
    probe_path.write_text(json.dumps(probe, indent=2), encoding="utf-8")
    print(f"[DONE] Training probe report -> {probe_path}")


if __name__ == "__main__":
    main()
