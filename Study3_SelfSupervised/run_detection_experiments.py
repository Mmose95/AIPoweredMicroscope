"""Run matched RF-DETR experiments with scratch, public, or own-data SSL DINOv3.

Open this file in PyCharm and press Run. It launches the configured paired
pilot without command-line arguments. Use ``--no-train`` only when you want to
audit paths and metadata without starting GPU training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rfdetr_dinov3_bridge import train_rfdetr_with_dinov3, write_bridge_provenance


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# PyCharm controls. Pressing Run launches this configured paired pilot.
# Command-line --no-train performs the same audit without starting training.
RUN_TRAINING = True
DEFAULT_ARMS = ("scratch", "public_ssl", "own_data_ssl", "public_domain_ssl")
DEFAULT_CONFIG = SCRIPT_DIR / "detection_experiment_config.json"
DEFAULT_DINOV3_REPO = REPO_ROOT.parent / "dinov3"
DEFAULT_DATASET_DIR = (
    REPO_ROOT
    / "SOLO_Supervised_RFDETR"
    / "Stat_Dataset40x"
    / "QA_40x-_20260901-135953"
)
DEFAULT_IMAGE_ROOT = Path(
    r"E:\PHD\PhdData\Patologi afd. - Aalborg\40x Input tiles for CVAT"
)
DEFAULT_SSL_CHECKPOINT = Path(
    r"E:\PHD\Results\SSL_QA40X\dinov3_vits16_full_seed0"
    r"\eval\training_81623\teacher_checkpoint.pth"
)
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "runs" / "detection_pilot"
VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _ucloud_member_root() -> Path | None:
    if os.name == "nt":
        return None
    work = Path("/work")
    if not work.is_dir():
        return None
    member = sorted(work.glob("Member Files:*"))
    if member:
        return member[0]
    hashed = sorted(path for path in work.glob("*#*") if path.is_dir())
    return hashed[0] if hashed else None


def _path_from_env(name: str, fallback: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else fallback


def _defaults() -> dict[str, Path]:
    member_root = _ucloud_member_root()
    image_fallback = (
        Path("/work/40x Input tiles for CVAT")
        if os.name != "nt" and Path("/work").is_dir()
        else DEFAULT_IMAGE_ROOT
    )
    output_fallback = (
        member_root / "DINOv3_Study3_OUTPUT" / "detection_experiments"
        if member_root
        else DEFAULT_OUTPUT_ROOT
    )
    checkpoint_fallback = (
        member_root / "DINOv3_Study3_OUTPUT" / "dinov3_vits16_full_seed0"
        / "eval" / "training_81623" / "teacher_checkpoint.pth"
        if member_root
        else DEFAULT_SSL_CHECKPOINT
    )
    return {
        "dinov3_repo": _path_from_env("DINOV3_REPO", DEFAULT_DINOV3_REPO),
        "dataset_dir": _path_from_env("STUDY3_DETECTION_DATASET", DEFAULT_DATASET_DIR),
        "image_root": _path_from_env("IMAGE_ROOT", image_fallback),
        "ssl_checkpoint": _path_from_env("STUDY3_SSL_CHECKPOINT", checkpoint_fallback),
        "public_ssl_weights": _path_from_env("STUDY3_PUBLIC_SSL_WEIGHTS", Path()),
        "output_root": _path_from_env("STUDY3_DETECTION_OUTPUT", output_fallback),
    }


def parse_args() -> argparse.Namespace:
    defaults = _defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--arms", nargs="+", choices=DEFAULT_ARMS, default=list(DEFAULT_ARMS))
    parser.add_argument("--dinov3-repo", type=Path, default=defaults["dinov3_repo"])
    parser.add_argument("--dataset-dir", type=Path, default=defaults["dataset_dir"])
    parser.add_argument("--image-root", type=Path, default=defaults["image_root"])
    parser.add_argument("--ssl-checkpoint", type=Path, default=defaults["ssl_checkpoint"])
    parser.add_argument("--public-ssl-weights", type=Path, default=defaults["public_ssl_weights"])
    parser.add_argument("--output-root", type=Path, default=defaults["output_root"])
    parser.add_argument("--train", dest="train", action="store_true", default=RUN_TRAINING)
    parser.add_argument("--no-train", dest="train", action="store_false")
    parser.add_argument("--overwrite-plan", action="store_true", default=False)
    return parser.parse_args()


def _load_coco(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"COCO annotation file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _image_index(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_relative: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES:
            resolved = path.resolve()
            by_relative[resolved.relative_to(root.resolve()).as_posix()] = resolved
            by_name.setdefault(path.name, []).append(resolved)
    return by_relative, by_name


def _resolve_image(
    file_name: str,
    image_root: Path,
    by_relative: dict[str, Path],
    by_name: dict[str, list[Path]],
) -> Path:
    candidate = Path(file_name)
    if candidate.is_absolute() and candidate.is_file():
        return candidate.resolve()
    normalized = file_name.replace("\\", "/").lstrip("./")
    direct = image_root / normalized
    if direct.is_file():
        return direct.resolve()
    indexed = by_relative.get(normalized)
    if indexed:
        return indexed
    matches = by_name.get(candidate.name, [])
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not uniquely resolve image {file_name!r} under {image_root}; "
        f"basename matches={len(matches)}"
    )


def _rank(image: dict, seed: int) -> str:
    token = f"{seed}:{image['id']}:{image['file_name']}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()


def _materialize_dataset(
    source: Path,
    destination: Path,
    image_root: Path,
    budget: float,
    seed: int,
) -> dict:
    if not (0.0 < budget <= 1.0):
        raise ValueError(f"Annotation budget must be in (0, 1], received {budget}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    # In the audited 40x COCO files, file_name is relative to image_root. Most
    # runs therefore avoid an expensive recursive inventory of the ~300 GB
    # image tree. Build that fallback index only if a direct path is missing.
    by_relative: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    fallback_index_built = False
    report: dict[str, Any] = {
        "source": str(source.resolve()),
        "image_root": str(image_root.resolve()),
        "annotation_budget": budget,
        "budget_unit": "annotated_training_images",
        "budget_seed": seed,
        "test_materialized": False,
        "splits": {},
    }
    for split in ("train", "valid"):
        source_json = source / split / "_annotations.coco.json"
        coco = _load_coco(source_json)
        original_images = list(coco.get("images", []))
        if split == "train" and budget < 1.0:
            keep_count = max(1, round(len(original_images) * budget))
            images = sorted(original_images, key=lambda item: _rank(item, seed))[:keep_count]
        else:
            images = original_images
        image_ids = {item["id"] for item in images}
        annotations = [a for a in coco.get("annotations", []) if a.get("image_id") in image_ids]
        for image in images:
            raw_name = str(image["file_name"])
            raw_path = Path(raw_name)
            direct_path = image_root / raw_name.replace("\\", "/").lstrip("./")
            if not (raw_path.is_absolute() and raw_path.is_file()) and not direct_path.is_file():
                if not fallback_index_built:
                    by_relative, by_name = _image_index(image_root)
                    fallback_index_built = True
            image["file_name"] = str(
                _resolve_image(raw_name, image_root, by_relative, by_name)
            )
        materialized = dict(coco)
        materialized["images"] = images
        materialized["annotations"] = annotations
        split_dir = destination / split
        split_dir.mkdir()
        output_json = split_dir / "_annotations.coco.json"
        _json_write(output_json, materialized)
        specimens = {
            Path(str(item["file_name"])).parent.name
            for item in images
        }
        report["splits"][split] = {
            "source_sha256": _sha256(source_json),
            "materialized_sha256": _sha256(output_json),
            "source_images": len(original_images),
            "images": len(images),
            "annotations": len(annotations),
            "specimens": len(specimens),
        }
    return report


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_config(config: dict) -> None:
    supported_arms = set(DEFAULT_ARMS)
    configured_arms = set(config.get("arms", []))
    if not configured_arms or not configured_arms <= supported_arms:
        raise ValueError(f"Config arms must be a non-empty subset of {sorted(supported_arms)}")
    if config.get("selection", {}).get("split") != "valid":
        raise ValueError("Checkpoint selection must use the validation split")
    if config.get("selection", {}).get("test_during_training") is not False:
        raise ValueError("test_during_training must be false")
    model = config.get("model", {})
    if model.get("architecture") != "dinov3_vits16" or int(model.get("patch_size", 0)) != 16:
        raise ValueError("This paired experiment requires dinov3_vits16 with patch_size=16")


def _run_one(
    *,
    arm: str,
    seed: int,
    budget: float,
    args: argparse.Namespace,
    config: dict,
) -> Path:
    budget_label = f"budget_{budget:.3f}".replace(".", "p")
    config_fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    run_dir = (
        args.output_root
        / config["experiment_name"]
        / f"config_{config_fingerprint}"
        / arm
        / f"seed_{seed}"
        / budget_label
    )
    if run_dir.exists() and any(run_dir.iterdir()) and not args.overwrite_plan:
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. Use --overwrite-plan only to replace an audit-only plan."
        )
    if run_dir.exists() and any(run_dir.iterdir()) and args.overwrite_plan:
        existing_record = run_dir / "run_record.json"
        existing_status = None
        if existing_record.is_file():
            existing_status = json.loads(existing_record.read_text(encoding="utf-8")).get("status")
        if existing_status != "planned":
            raise RuntimeError(
                "--overwrite-plan can replace only a prior audit plan; "
                f"the existing run status is {existing_status!r}"
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = run_dir / "effective_dataset"
    dataset_report = _materialize_dataset(
        args.dataset_dir.resolve(), dataset_dir, args.image_root.resolve(), budget, seed
    )
    checkpoint = args.ssl_checkpoint.resolve() if arm in ("own_data_ssl", "public_domain_ssl") else None
    public_weights = args.public_ssl_weights.resolve() if arm in ("public_ssl", "public_domain_ssl") else None
    if arm in ("own_data_ssl", "public_domain_ssl") and not checkpoint.is_file():
        raise FileNotFoundError(f"SSL teacher checkpoint not found: {checkpoint}")
    if arm in ("public_ssl", "public_domain_ssl") and not public_weights.is_file():
        raise FileNotFoundError(
            "Official public DINOv3 weights not found. Request Meta access, download "
            "dinov3_vits16_pretrain_lvd1689m-08c60483.pth, then pass --public-ssl-weights. "
            f"Received: {public_weights}"
        )
    if arm == "scratch" and checkpoint is not None:
        raise AssertionError("Scratch arm unexpectedly received a checkpoint")

    run_record = {
        "status": "planned" if not args.train else "starting",
        "created_utc": _utc_now(),
        "arm": arm,
        "seed": seed,
        "annotation_budget": budget,
        "configuration": config,
        "configuration_fingerprint": config_fingerprint,
        "paths": {
            "source_dataset": str(args.dataset_dir.resolve()),
            "effective_dataset": str(dataset_dir.resolve()),
            "image_root": str(args.image_root.resolve()),
            "dinov3_repo": str(args.dinov3_repo.resolve()),
            "output": str(run_dir.resolve()),
        },
        "initialization": {
            "backbone": {
                "scratch": "random",
                "public_ssl": "official_dinov3_vits16_lvd1689m",
                "own_data_ssl": "own_data_ssl_ema_teacher",
                "public_domain_ssl": "official_dinov3_then_domain_ssl_ema_teacher",
            }[arm],
            "detector": "random",
            "external_pretrained_weights": arm in ("public_ssl", "public_domain_ssl"),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_sha256": _sha256(checkpoint) if checkpoint else None,
            "public_weights": str(public_weights) if public_weights else None,
            "public_weights_sha256": _sha256(public_weights) if public_weights else None,
        },
        "dataset": dataset_report,
        "test_policy": "Test split is not materialized and run_test=False.",
    }
    record_path = run_dir / "run_record.json"
    _json_write(record_path, run_record)
    print(json.dumps(run_record, indent=2))
    if not args.train:
        return run_dir

    try:
        from rfdetr import RFDETRSmall

        _seed_everything(seed)
        model_config = config["model"]
        training = config["training"]
        rf_model = RFDETRSmall(
            pretrain_weights=None,
            resolution=int(model_config["resolution"]),
            patch_size=int(model_config["patch_size"]),
            num_queries=int(model_config["num_queries"]),
            gradient_checkpointing=bool(training["gradient_checkpointing"]),
        )
        encoder = train_rfdetr_with_dinov3(
            rf_model,
            dinov3_repo=args.dinov3_repo.resolve(),
            initialization=arm,
            checkpoint=checkpoint,
            public_weights=public_weights,
            architecture=model_config["architecture"],
            feature_layers=tuple(model_config["feature_layers"]),
            dataset_dir=str(dataset_dir.resolve()),
            output_dir=str(run_dir.resolve()),
            class_names=list(model_config["class_names"]),
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            grad_accum_steps=int(training["grad_accum_steps"]),
            lr=float(training["lr"]),
            weight_decay=float(training["weight_decay"]),
            num_workers=int(training["num_workers"]),
            checkpoint_interval=int(training["checkpoint_interval"]),
            seed=seed,
            early_stopping=False,
            run_test=False,
        )
        write_bridge_provenance(encoder, run_dir / "backbone_provenance.json")
        run_record["backbone_provenance"] = encoder.provenance_dict()
        run_record["status"] = "completed"
        run_record["completed_utc"] = _utc_now()
    except BaseException as exc:
        run_record["status"] = "failed"
        run_record["failed_utc"] = _utc_now()
        run_record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        (run_dir / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        _json_write(record_path, run_record)
    return run_dir


def main() -> int:
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    for required in (args.dinov3_repo / "hubconf.py", args.dataset_dir, args.image_root):
        if not required.exists():
            raise FileNotFoundError(f"Required path not found: {required}")
    requested_arms = list(args.arms)
    if any(arm not in config["arms"] for arm in requested_arms):
        raise ValueError("Requested arm is absent from the experiment configuration")
    completed = []
    for seed in config["seeds"]:
        for budget in config["annotation_budgets"]:
            for arm in requested_arms:
                completed.append(
                    str(
                        _run_one(
                            arm=arm,
                            seed=int(seed),
                            budget=float(budget),
                            args=args,
                            config=config,
                        )
                    )
                )
    print(json.dumps({"status": "completed" if args.train else "audit_passed", "runs": completed}, indent=2))
    if not args.train:
        print("Audit only: run without --no-train to begin the configured paired pilot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
