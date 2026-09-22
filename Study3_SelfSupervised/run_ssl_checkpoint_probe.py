"""Rank DINOv3 SSL teacher checkpoints with a frozen-backbone detection probe.

Open this file in PyCharm and press Run to print the planned checkpoints without
starting training. Set ``RUN_PROBES = True`` below, or pass ``--run`` on UCloud,
to train identical RF-DETR Small heads while keeping every DINOv3 backbone
strictly frozen.

The default strategy probes SSL epoch 1, every fifth SSL epoch, and the final
available checkpoint. It then probes the two neighboring SSL epochs on either
side of the best coarse checkpoint. Checkpoint ranking uses validation EMA
mAP@50:95; the test split is never materialized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "ssl_checkpoint_probe_config.json"
DEFAULT_DINOV3_REPO = PROJECT_DIR.parent / "dinov3"
DEFAULT_DATASET_DIR = (
    PROJECT_DIR
    / "SOLO_Supervised_RFDETR"
    / "Stat_Dataset40x"
    / "QA_40x-_20260901-135953"
)
DEFAULT_IMAGE_ROOT = Path(r"E:\PHD\PhdData\Patologi afd. - Aalborg\40x Input tiles for CVAT")
DEFAULT_SSL_RUN_DIR = Path(r"E:\PHD\Results\SSL_QA40X\dinov3_vits16_full_seed0")
DEFAULT_OUTPUT_ROOT = Path(r"E:\PHD\Results\SSL_QA40X\SSL_Checkpoint_Probe")

# PyCharm control. Keep False for a quick plan; set True to launch local probes.
RUN_PROBES = False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else fallback


def _defaults() -> dict[str, Path]:
    ucloud = os.name != "nt" and Path("/work").is_dir()
    output_base = _env_path("OUTPUT_ROOT", Path("/work/DINOv3_Study3_OUTPUT")) if ucloud else None
    return {
        "dinov3_repo": _env_path("DINOV3_REPO", Path("/work/projects/dinov3") if ucloud else DEFAULT_DINOV3_REPO),
        "dataset_dir": _env_path("STUDY3_DETECTION_DATASET", Path("/work/projects/myproj/SOLO_Supervised_RFDETR/Stat_Dataset40x/QA_40x-_20260901-135953") if ucloud else DEFAULT_DATASET_DIR),
        "image_root": _env_path("IMAGE_ROOT", Path("/work/40x Input tiles for CVAT") if ucloud else DEFAULT_IMAGE_ROOT),
        "ssl_run_dir": _env_path("STUDY3_SSL_RUN_DIR", output_base / "dinov3_vits16_full_seed0" if ucloud else DEFAULT_SSL_RUN_DIR),
        "output_root": _env_path("STUDY3_SSL_PROBE_OUTPUT", output_base / "SSL_Checkpoint_Probe" if ucloud else DEFAULT_OUTPUT_ROOT),
    }


def parse_args() -> argparse.Namespace:
    defaults = _defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dinov3-repo", type=Path, default=defaults["dinov3_repo"])
    parser.add_argument("--dataset-dir", type=Path, default=defaults["dataset_dir"])
    parser.add_argument("--image-root", type=Path, default=defaults["image_root"])
    parser.add_argument("--ssl-run-dir", type=Path, default=defaults["ssl_run_dir"])
    parser.add_argument("--output-root", type=Path, default=defaults["output_root"])
    parser.add_argument("--stride-epochs", type=int, default=5)
    parser.add_argument("--fine-radius-epochs", type=int, default=2)
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="*",
        help="Probe only these exact training_<iteration> checkpoints and skip automatic refinement.",
    )
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--run", dest="run", action="store_true", default=RUN_PROBES)
    parser.add_argument("--plan-only", dest="run", action="store_false")
    return parser.parse_args()


def _discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for checkpoint in (run_dir / "eval").glob("training_*/teacher_checkpoint.pth"):
        try:
            iteration = int(checkpoint.parent.name.removeprefix("training_"))
        except ValueError:
            continue
        checkpoints.append((iteration, checkpoint.resolve()))
    checkpoints.sort()
    if not checkpoints:
        raise FileNotFoundError(f"No teacher checkpoints found under {run_dir / 'eval'}")
    return checkpoints


def _epoch_length(run_dir: Path) -> int:
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"SSL config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    value = int(config["train"]["OFFICIAL_EPOCH_LENGTH"])
    if value <= 0:
        raise ValueError(f"Invalid OFFICIAL_EPOCH_LENGTH={value}")
    return value


def _ssl_epoch(iteration: int, epoch_length: int) -> float:
    return (iteration + 1) / epoch_length


def _coarse_checkpoints(
    checkpoints: list[tuple[int, Path]],
    epoch_length: int,
    stride_epochs: int,
) -> list[tuple[int, Path]]:
    if stride_epochs <= 0:
        raise ValueError("--stride-epochs must be positive")
    selected: list[tuple[int, Path]] = []
    final_iteration = checkpoints[-1][0]
    for index, (iteration, checkpoint) in enumerate(checkpoints):
        epoch = _ssl_epoch(iteration, epoch_length)
        rounded = round(epoch)
        is_integer_epoch = abs(epoch - rounded) < 1e-6
        if index == 0 or iteration == final_iteration or (is_integer_epoch and rounded % stride_epochs == 0):
            selected.append((iteration, checkpoint))
    return selected


def _config_fingerprint(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]


def _expected_run_dir(checkpoint_output: Path, config: dict[str, Any]) -> Path:
    return (
        checkpoint_output
        / config["experiment_name"]
        / f"config_{_config_fingerprint(config)}"
        / "own_data_ssl"
        / "seed_0"
        / "budget_1p000"
    )


def _run_status(run_dir: Path) -> str | None:
    record = run_dir / "run_record.json"
    if not record.is_file():
        return None
    return json.loads(record.read_text(encoding="utf-8")).get("status")


def _launch_probe(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    iteration: int,
    checkpoint: Path,
) -> Path:
    checkpoint_output = args.output_root / f"ssl_iter_{iteration:06d}"
    run_dir = _expected_run_dir(checkpoint_output, config)
    status = _run_status(run_dir)
    if status == "completed":
        print(f"[Probe] iteration {iteration}: already completed")
        return run_dir

    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_detection_experiments.py"),
        "--config", str(args.config.resolve()),
        "--arms", "own_data_ssl",
        "--dinov3-repo", str(args.dinov3_repo.resolve()),
        "--dataset-dir", str(args.dataset_dir.resolve()),
        "--image-root", str(args.image_root.resolve()),
        "--ssl-checkpoint", str(checkpoint),
        "--output-root", str(checkpoint_output.resolve()),
        "--train",
    ]
    last_checkpoint = run_dir / "last.ckpt"
    if status in {"starting", "resuming", "failed"} and last_checkpoint.is_file():
        command.extend(["--resume", str(last_checkpoint.resolve())])
        print(f"[Probe] iteration {iteration}: resuming interrupted detector probe")
    elif status is not None:
        raise RuntimeError(
            f"Probe iteration {iteration} has status {status!r} but no resumable last.ckpt: {run_dir}"
        )
    else:
        print(f"[Probe] iteration {iteration}: starting frozen-backbone detector probe")
    subprocess.run(command, check=True)
    if _run_status(run_dir) != "completed":
        raise RuntimeError(f"Probe did not complete cleanly: {run_dir}")
    return run_dir


def _best_validation(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("val/ema_mAP_50_95")]
    if not rows:
        raise RuntimeError(f"No validation metrics in {metrics_path}")
    best = max(
        rows,
        key=lambda row: (float(row["val/ema_mAP_50_95"]), float(row["val/ema_mAP_50"])),
    )
    return {
        "probe_best_detector_epoch": int(float(best["epoch"])),
        "ema_mAP_50_95": float(best["val/ema_mAP_50_95"]),
        "ema_mAP_50": float(best["val/ema_mAP_50"]),
        "ema_mAR": float(best["val/ema_mAR"]),
        "regular_mAP_50_95": float(best["val/mAP_50_95"]),
        "regular_mAP_50": float(best["val/mAP_50"]),
    }


def _collect_result(
    iteration: int,
    checkpoint: Path,
    epoch_length: int,
    run_dir: Path,
    stage: str,
) -> dict[str, Any]:
    return {
        "ssl_iteration": iteration,
        "ssl_epoch": _ssl_epoch(iteration, epoch_length),
        "selection_stage": stage,
        "teacher_checkpoint": str(checkpoint),
        "teacher_checkpoint_sha256": _sha256(checkpoint),
        "probe_run_dir": str(run_dir.resolve()),
        **_best_validation(run_dir),
    }


def _write_summary(output_root: Path, results: list[dict[str, Any]], plan: dict[str, Any]) -> None:
    ordered = sorted(results, key=lambda item: item["ssl_iteration"])
    ranked = sorted(
        ordered,
        key=lambda item: (item["ema_mAP_50_95"], item["ema_mAP_50"]),
        reverse=True,
    )
    payload = {
        "status": "completed",
        "updated_utc": _utc_now(),
        "selection_rule": "maximum validation EMA mAP@50:95; EMA mAP@50 tie-break",
        "test_used": False,
        "best_checkpoint": ranked[0] if ranked else None,
        "ranked_results": ranked,
        "trajectory_results": ordered,
        "plan": plan,
    }
    _json_write(output_root / "probe_summary.json", payload)
    if ordered:
        columns = list(ordered[0])
        csv_path = output_root / "probe_summary.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(ordered)
        temporary.replace(csv_path)


def _fine_candidates(
    all_checkpoints: list[tuple[int, Path]],
    epoch_length: int,
    winner_iteration: int,
    radius_epochs: int,
    completed_iterations: set[int],
) -> list[tuple[int, Path]]:
    if radius_epochs <= 0:
        return []
    winner_epoch = _ssl_epoch(winner_iteration, epoch_length)
    return [
        item
        for item in all_checkpoints
        if item[0] not in completed_iterations
        and abs(_ssl_epoch(item[0], epoch_length) - winner_epoch) <= radius_epochs
    ]


def main() -> int:
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.ssl_run_dir = args.ssl_run_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("model", {}).get("freeze_encoder") is not True:
        raise ValueError("Probe config must set model.freeze_encoder=true")
    if config.get("selection", {}).get("test_during_training") is not False:
        raise ValueError("Probe config must keep the test set unused")

    all_checkpoints = _discover_checkpoints(args.ssl_run_dir)
    epoch_length = _epoch_length(args.ssl_run_dir)
    by_iteration = dict(all_checkpoints)
    if args.iterations is not None:
        missing = sorted(set(args.iterations) - set(by_iteration))
        if missing:
            raise FileNotFoundError(f"Requested SSL iterations are unavailable: {missing}")
        coarse = [(iteration, by_iteration[iteration]) for iteration in sorted(set(args.iterations))]
    else:
        coarse = _coarse_checkpoints(all_checkpoints, epoch_length, args.stride_epochs)
    if args.max_checkpoints is not None:
        coarse = coarse[: args.max_checkpoints]

    plan = {
        "created_utc": _utc_now(),
        "ssl_run_dir": str(args.ssl_run_dir),
        "ssl_epoch_length_iterations": epoch_length,
        "available_checkpoint_count": len(all_checkpoints),
        "coarse_stride_ssl_epochs": args.stride_epochs,
        "fine_radius_ssl_epochs": 0 if args.iterations is not None else args.fine_radius_epochs,
        "probe_detector_epochs": int(config["training"]["epochs"]),
        "frozen_backbone": True,
        "coarse_checkpoints": [
            {"iteration": iteration, "ssl_epoch": _ssl_epoch(iteration, epoch_length), "path": str(path)}
            for iteration, path in coarse
        ],
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    _json_write(args.output_root / "probe_plan.json", plan)
    print(json.dumps(plan, indent=2))
    if not args.run:
        print("[Probe] Plan only. Pass --run or set RUN_PROBES=True to launch training.")
        return 0

    results: list[dict[str, Any]] = []
    for iteration, checkpoint in coarse:
        run_dir = _launch_probe(
            args=args,
            config=config,
            iteration=iteration,
            checkpoint=checkpoint,
        )
        results.append(_collect_result(iteration, checkpoint, epoch_length, run_dir, "coarse"))
        _write_summary(args.output_root, results, plan)

    if args.iterations is None and results:
        coarse_winner = max(
            results,
            key=lambda item: (item["ema_mAP_50_95"], item["ema_mAP_50"]),
        )
        completed = {item["ssl_iteration"] for item in results}
        fine = _fine_candidates(
            all_checkpoints,
            epoch_length,
            int(coarse_winner["ssl_iteration"]),
            args.fine_radius_epochs,
            completed,
        )
        plan["coarse_winner_iteration"] = coarse_winner["ssl_iteration"]
        plan["fine_checkpoints"] = [
            {"iteration": iteration, "ssl_epoch": _ssl_epoch(iteration, epoch_length), "path": str(path)}
            for iteration, path in fine
        ]
        _json_write(args.output_root / "probe_plan.json", plan)
        for iteration, checkpoint in fine:
            run_dir = _launch_probe(
                args=args,
                config=config,
                iteration=iteration,
                checkpoint=checkpoint,
            )
            results.append(_collect_result(iteration, checkpoint, epoch_length, run_dir, "fine"))
            _write_summary(args.output_root, results, plan)

    _write_summary(args.output_root, results, plan)
    best = max(results, key=lambda item: (item["ema_mAP_50_95"], item["ema_mAP_50"]))
    print(json.dumps({"status": "completed", "best_checkpoint": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
