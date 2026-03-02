from __future__ import annotations

import csv
import glob
import json
import os
import os.path as op
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path


VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
WINDOWS_PATH_RE = re.compile(r"^[a-zA-Z]:\\")


def _main_print(*args, **kwargs):
    print(*args, **kwargs)


def _detect_runtime_profile() -> str:
    forced = os.environ.get("DETREG_RUNTIME_PROFILE", "auto").strip().lower()
    if forced in {"ucloud", "local"}:
        return forced
    if forced not in {"", "auto"}:
        raise ValueError("DETREG_RUNTIME_PROFILE must be one of: auto, ucloud, local")
    if os.name != "nt" and Path("/work").exists():
        has_member_files = any(Path("/work").glob("Member Files:*"))
        has_hash_user = any(Path("/work").glob("*#*"))
        if has_member_files or has_hash_user:
            return "ucloud"
    return "local"


def _detect_ucloud_user_base() -> str:
    work = Path("/work")
    if not work.exists():
        return ""
    member = sorted(work.glob("Member Files:*"))
    if member:
        return member[0].name
    hashed = sorted([p for p in work.glob("*#*") if p.is_dir()])
    return hashed[0].name if hashed else ""


def _csv_tokens(text: str) -> list[str]:
    return [t.strip() for t in str(text).split(",") if t.strip()]


def _csv_float(text: str) -> list[float]:
    vals = [float(t) for t in _csv_tokens(text)]
    if not vals:
        raise ValueError("Expected at least one float value")
    for v in vals:
        if v <= 0 or v > 1:
            raise ValueError(f"Train fraction must be in (0,1], got {v}")
    return vals


def _csv_int(text: str) -> list[int]:
    vals = [int(t) for t in _csv_tokens(text)]
    if not vals:
        raise ValueError("Expected at least one integer value")
    return vals


def _index_image_paths(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_rel: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    root = root.resolve()
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VALID_EXTS:
            continue
        rel = p.resolve().relative_to(root).as_posix()
        by_rel[rel] = p
        by_name.setdefault(p.name, []).append(p)
    return by_rel, by_name


def _resolve_image_path(file_name: str, images_root: Path, by_rel: dict[str, Path], by_name: dict[str, list[Path]]) -> Path:
    rel = file_name.replace("\\", "/")

    direct = images_root / rel
    if direct.exists():
        return direct

    if rel in by_rel:
        return by_rel[rel]

    base = Path(rel).name
    if base in by_name:
        cands = sorted(by_name[base], key=lambda x: len(str(x)))
        return cands[0]

    if WINDOWS_PATH_RE.match(file_name) and base in by_name:
        cands = sorted(by_name[base], key=lambda x: len(str(x)))
        return cands[0]

    raise FileNotFoundError(f"Could not resolve image '{file_name}' under {images_root}")


def _load_coco_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing COCO JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _normalize_single_class(coco: dict, class_name: str) -> dict:
    images = list(coco.get("images", []))
    anns = list(coco.get("annotations", []))
    new_anns = []
    for i, ann in enumerate(anns, start=1):
        a = dict(ann)
        a["id"] = i
        a["category_id"] = 1
        if "area" not in a or a["area"] is None:
            x, y, w, h = a.get("bbox", [0.0, 0.0, 0.0, 0.0])
            a["area"] = float(max(0.0, w) * max(0.0, h))
        if "iscrowd" not in a:
            a["iscrowd"] = 0
        new_anns.append(a)
    return {
        "images": images,
        "annotations": new_anns,
        "categories": [{"id": 1, "name": class_name, "supercategory": "cell"}],
    }


def _subset_train_split(coco_train: dict, frac: float, seed: int) -> dict:
    images = list(coco_train.get("images", []))
    anns = list(coco_train.get("annotations", []))
    cats = list(coco_train.get("categories", []))

    if frac >= 0.999:
        return {"images": images, "annotations": anns, "categories": cats}

    n_keep = max(1, int(round(len(images) * frac)))
    idxs = list(range(len(images)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    keep_set = set(idxs[:n_keep])

    kept_images = [im for i, im in enumerate(images) if i in keep_set]
    kept_ids = {im["id"] for im in kept_images}
    kept_anns = [a for a in anns if a["image_id"] in kept_ids]
    return {"images": kept_images, "annotations": kept_anns, "categories": cats}


def _materialize_image(src: Path, dst: Path, link_mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link_mode == "copy":
        shutil.copy2(src, dst)
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _materialize_split_images(images: list[dict], split_img_dir: Path, link_mode: str) -> tuple[list[dict], int, int]:
    remapped = []
    n_links = 0
    n_copies = 0
    for im in images:
        src_path = Path(im["file_name"])
        ext = src_path.suffix if src_path.suffix else ".png"
        out_name = f"{int(im['id']):012d}{ext.lower()}"
        out_path = split_img_dir / out_name
        before_exists = out_path.exists()
        _materialize_image(src_path, out_path, link_mode=link_mode)
        if not before_exists:
            if out_path.is_symlink():
                n_links += 1
            else:
                n_copies += 1
        im2 = dict(im)
        im2["file_name"] = out_name
        remapped.append(im2)
    return remapped, n_links, n_copies


def _parse_detreg_metrics(log_path: Path) -> dict:
    if not log_path.exists():
        return {
            "metrics_source": "missing_log",
            "best_epoch": None,
            "val_AP50": None,
            "val_mAP5095": None,
            "val_AP75": None,
        }

    best = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        stats = rec.get("test_coco_eval_bbox")
        if not isinstance(stats, list) or len(stats) < 3:
            continue
        m = {
            "epoch": rec.get("epoch"),
            "ap5095": float(stats[0]),
            "ap50": float(stats[1]),
            "ap75": float(stats[2]),
        }
        if best is None or m["ap50"] > best["ap50"]:
            best = m

    if best is None:
        return {
            "metrics_source": "log_no_eval_rows",
            "best_epoch": None,
            "val_AP50": None,
            "val_mAP5095": None,
            "val_AP75": None,
        }

    return {
        "metrics_source": "log.txt",
        "best_epoch": best["epoch"],
        "val_AP50": best["ap50"],
        "val_mAP5095": best["ap5095"],
        "val_AP75": best["ap75"],
    }


def _download_if_needed(path: Path, url: str):
    if path.exists():
        return
    if not url:
        raise FileNotFoundError(
            f"DETReg pretrain checkpoint missing at {path} and no DETREG_PRETRAIN_URL provided."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    _main_print(f"[PRETRAIN] Downloading DETReg checkpoint:\n  url={url}\n  dst={path}")
    urllib.request.urlretrieve(url, path)
    _main_print(f"[PRETRAIN] Download complete ({path.stat().st_size} bytes)")


def _latest_dataset_dir(root: Path, token: str) -> Path | None:
    cands = sorted([p for p in root.glob(f"*{token}*") if p.is_dir()])
    return cands[-1] if cands else None


def _resolve_runtime_defaults() -> dict:
    runtime_profile = _detect_runtime_profile()
    user_base = _detect_ucloud_user_base() if runtime_profile == "ucloud" else ""

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    if runtime_profile == "ucloud":
        default_project_dir = Path("/work/projects/myproj/SOLO_Supervised_RFDETR")
        detreg_repo_default = Path("/work/projects/myproj/DETReg")
        datasets_root_default = default_project_dir / "Stat_Dataset"
        if user_base:
            images_root_default = Path("/work") / user_base / "CellScanData" / "Zoom10x - Quality Assessment_Cleaned"
            output_root_default = Path("/work") / user_base / "RFDETR_SOLO_OUTPUT" / "DETREG_EPI"
            pretrain_ckpt_default = Path("/work") / user_base / "DETReg_Checkpoints" / "checkpoint_coco.pth"
        else:
            images_root_default = repo_root / "CellScanData" / "Zoom10x - Quality Assessment_Cleaned"
            output_root_default = default_project_dir / "RFDETR_SOLO_OUTPUT" / "DETREG_EPI"
            pretrain_ckpt_default = repo_root / "DETReg_Checkpoints" / "checkpoint_coco.pth"
    else:
        default_project_dir = script_dir
        detreg_repo_default = repo_root / "_external_DETReg_ref"
        datasets_root_default = script_dir / "Stat_Dataset"
        images_root_default = Path(r"D:\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment_Cleaned")
        output_root_default = script_dir / "RFDETR_SOLO_OUTPUT" / "DETREG_EPI_local"
        pretrain_ckpt_default = repo_root / "DETReg_Checkpoints" / "checkpoint_coco.pth"

    return {
        "runtime_profile": runtime_profile,
        "user_base": user_base,
        "project_dir": Path(os.environ.get("PROJECT_DIR", str(default_project_dir))),
        "detreg_repo_dir": Path(os.environ.get("DETREG_REPO_DIR", str(detreg_repo_default))),
        "datasets_root": Path(os.environ.get("STAT_DATASETS_ROOT", str(datasets_root_default))),
        "images_root": Path(os.environ.get("IMAGES_FALLBACK_ROOT", str(images_root_default))),
        "output_root": Path(os.environ.get("OUTPUT_ROOT", str(output_root_default))),
        "pretrain_ckpt": Path(os.environ.get("DETREG_PRETRAIN_CKPT", str(pretrain_ckpt_default))),
        "pretrain_url": os.environ.get(
            "DETREG_PRETRAIN_URL",
            "https://github.com/amirbar/DETReg/releases/download/1.0.0/checkpoint_coco.pth",
        ).strip(),
    }


def _build_resolved_dataset(src_dir: Path, dst_dir: Path, images_root: Path) -> Path:
    ok_marker = dst_dir / ".RESOLVED_OK"
    if ok_marker.exists():
        _main_print(f"[RESOLVE] Using cached resolved dataset: {dst_dir}")
        return dst_dir

    if not images_root.exists():
        raise FileNotFoundError(f"IMAGES_FALLBACK_ROOT does not exist: {images_root}")

    by_rel, by_name = _index_image_paths(images_root)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        src_json = src_dir / split / "_annotations.coco.json"
        if not src_json.exists():
            continue
        data = _load_coco_json(src_json)
        out_images = []
        missing = 0
        for im in data.get("images", []):
            try:
                resolved = _resolve_image_path(im["file_name"], images_root, by_rel, by_name)
            except FileNotFoundError as e:
                _main_print(f"[RESOLVE][WARN] {e}")
                missing += 1
                continue
            im2 = dict(im)
            im2["file_name"] = str(resolved.resolve())
            out_images.append(im2)

        valid_ids = {im["id"] for im in out_images}
        out_anns = [a for a in data.get("annotations", []) if a["image_id"] in valid_ids]
        out_data = {
            "images": out_images,
            "annotations": out_anns,
            "categories": data.get("categories", []),
        }
        out_split = dst_dir / split
        out_split.mkdir(parents=True, exist_ok=True)
        _write_json(out_split / "_annotations.coco.json", out_data)
        _main_print(
            f"[RESOLVE] {split}: kept {len(out_images)} images, {len(out_anns)} anns (missing={missing})"
        )

    ok_marker.write_text("ok", encoding="utf-8")
    return dst_dir


def _build_detreg_coco_data_root(
    resolved_dataset_dir: Path,
    cache_root: Path,
    class_name: str,
    train_fraction: float,
    fraction_seed: int,
    link_mode: str,
) -> Path:
    frac_txt = f"{float(train_fraction):.5f}".rstrip("0").rstrip(".").replace(".", "p")
    cache_dir = cache_root / f"{resolved_dataset_dir.name}_detreg_frac_{frac_txt}_fseed{fraction_seed}_{link_mode}"
    ok_marker = cache_dir / ".DETREG_DATA_OK"
    manifest_path = cache_dir / "detreg_dataset_manifest.json"
    if ok_marker.exists() and manifest_path.exists():
        _main_print(f"[DETREG DATA] Using cached data root: {cache_dir}")
        return cache_dir

    coco_root = cache_dir / "MSCoco"
    train_img_dir = coco_root / "train2017"
    val_img_dir = coco_root / "val2017"
    ann_dir = coco_root / "annotations"
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    train_data = _load_coco_json(resolved_dataset_dir / "train" / "_annotations.coco.json")
    val_data = _load_coco_json(resolved_dataset_dir / "valid" / "_annotations.coco.json")

    train_subset = _subset_train_split(train_data, frac=train_fraction, seed=fraction_seed)
    train_subset = _normalize_single_class(train_subset, class_name=class_name)
    val_subset = _normalize_single_class(val_data, class_name=class_name)

    remapped_train_images, train_links, train_copies = _materialize_split_images(
        train_subset["images"], split_img_dir=train_img_dir, link_mode=link_mode
    )
    remapped_val_images, val_links, val_copies = _materialize_split_images(
        val_subset["images"], split_img_dir=val_img_dir, link_mode=link_mode
    )

    train_out = {
        "images": remapped_train_images,
        "annotations": train_subset["annotations"],
        "categories": train_subset["categories"],
    }
    val_out = {
        "images": remapped_val_images,
        "annotations": val_subset["annotations"],
        "categories": val_subset["categories"],
    }

    _write_json(ann_dir / "instances_train2017.json", train_out)
    _write_json(ann_dir / "instances_val2017.json", val_out)

    manifest = {
        "source_dataset": str(resolved_dataset_dir),
        "class_name": class_name,
        "train_fraction": float(train_fraction),
        "fraction_seed": int(fraction_seed),
        "link_mode": link_mode,
        "train_images": len(train_out["images"]),
        "train_annotations": len(train_out["annotations"]),
        "val_images": len(val_out["images"]),
        "val_annotations": len(val_out["annotations"]),
        "train_links": int(train_links),
        "train_copies": int(train_copies),
        "val_links": int(val_links),
        "val_copies": int(val_copies),
    }
    _write_json(manifest_path, manifest)
    ok_marker.write_text("ok", encoding="utf-8")
    _main_print(
        f"[DETREG DATA] Built {cache_dir}: "
        f"train={manifest['train_images']} imgs / {manifest['train_annotations']} anns, "
        f"val={manifest['val_images']} imgs / {manifest['val_annotations']} anns"
    )
    return cache_dir


def _run_and_stream(cmd: list[str], cwd: Path, env: dict[str, str]):
    _main_print("[LAUNCH]")
    _main_print(" cwd:", str(cwd))
    _main_print(" cmd:", " ".join(shlex.quote(x) for x in cmd))

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def _preflight_detreg_runtime(detreg_repo_dir: Path, python_exec: str):
    probe = [
        python_exec,
        "-c",
        (
            "import torch; import pycocotools; "
            "from models.ops.modules import MSDeformAttn; "
            "print('DETREG_RUNTIME_OK')"
        ),
    ]
    proc = subprocess.run(
        probe,
        cwd=str(detreg_repo_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        raise RuntimeError(
            "DETReg runtime preflight failed. Ensure DETReg dependencies are installed and CUDA ops are built.\n"
            "Expected setup inside DETREG_REPO_DIR:\n"
            "  pip install -r requirements.txt\n"
            "  cd models/ops && sh make.sh && python test.py\n"
            f"Probe output:\n{out}"
        )
    _main_print("[DETREG] Runtime preflight OK")


def _write_training_plan(session_dir: Path, rows: list[dict]):
    json_path = session_dir / "TRAINING_PLAN.json"
    csv_path = session_dir / "TRAINING_PLAN.csv"
    _write_json(json_path, {"rows": rows})

    fieldnames = [
        "run_idx",
        "target",
        "init_mode",
        "train_fraction",
        "fraction_seed",
        "seed",
        "epochs",
        "batch_size",
        "lr",
        "lr_drop",
        "output_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in fieldnames})


def _write_run_summary_csv(session_dir: Path, rows: list[dict]):
    out = session_dir / "RUN_SUMMARY.csv"
    fieldnames = [
        "run_idx",
        "target",
        "init_mode",
        "train_fraction",
        "fraction_seed",
        "seed",
        "val_AP50",
        "val_mAP5095",
        "val_AP75",
        "best_epoch",
        "metrics_source",
        "seconds",
        "output_dir",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in fieldnames})


def main():
    cfg = _resolve_runtime_defaults()
    runtime_profile = cfg["runtime_profile"]
    user_base = cfg["user_base"]
    project_dir = Path(cfg["project_dir"]).resolve()
    detreg_repo_dir = Path(cfg["detreg_repo_dir"]).resolve()
    datasets_root = Path(cfg["datasets_root"]).resolve()
    images_root = Path(cfg["images_root"]).resolve()
    output_root = Path(cfg["output_root"]).resolve()
    pretrain_ckpt = Path(cfg["pretrain_ckpt"]).resolve()
    pretrain_url = cfg["pretrain_url"]

    target_name = os.environ.get("DETREG_TARGET_NAME", "Squamous Epithelial Cell").strip()
    train_fractions = _csv_float(os.environ.get("DETREG_TRAIN_FRACTIONS", "0.5,1.0"))
    seeds = _csv_int(os.environ.get("DETREG_SEEDS", "42,43,44"))
    fraction_seed = int(os.environ.get("DETREG_FRACTION_SEED", "42"))

    detreg_model = os.environ.get("DETREG_MODEL", "deformable_detr").strip()
    epochs = int(os.environ.get("DETREG_EPOCHS", "50"))
    batch_size = int(os.environ.get("DETREG_BATCH_SIZE", "4"))
    lr_drop = int(os.environ.get("DETREG_LR_DROP", "40"))
    eval_every = int(os.environ.get("DETREG_EVAL_EVERY", "1"))
    num_workers = int(os.environ.get("DETREG_NUM_WORKERS", "8"))
    lr_text = os.environ.get("DETREG_LR", "").strip()
    link_mode = os.environ.get("DETREG_DATA_LINK_MODE", "symlink").strip().lower()
    if link_mode not in {"symlink", "copy"}:
        raise ValueError("DETREG_DATA_LINK_MODE must be 'symlink' or 'copy'")
    auto_download = os.environ.get("DETREG_AUTO_DOWNLOAD_PRETRAIN", "1").strip() in {"1", "true", "True"}
    extra_args = shlex.split(os.environ.get("DETREG_EXTRA_ARGS", ""))
    python_exec = os.environ.get("DETREG_PYTHON", sys.executable).strip() or sys.executable

    dataset_epi = os.environ.get("DATASET_EPI", "").strip()
    if not dataset_epi:
        latest = _latest_dataset_dir(datasets_root, "SquamousEpithelialCell_OVR")
        if latest is None:
            raise FileNotFoundError(
                f"Could not auto-detect epithelial dataset under {datasets_root}. "
                "Set DATASET_EPI explicitly."
            )
        dataset_epi = str(latest)
    dataset_epi_path = Path(dataset_epi).resolve()

    _main_print("DETREG_RUNTIME_PROFILE =", runtime_profile)
    _main_print("USER_BASE_DIR =", user_base or "<none>")
    _main_print("PROJECT_DIR =", str(project_dir))
    _main_print("DETREG_REPO_DIR =", str(detreg_repo_dir))
    _main_print("STAT_DATASETS_ROOT =", str(datasets_root))
    _main_print("DATASET_EPI =", str(dataset_epi_path))
    _main_print("IMAGES_FALLBACK_ROOT =", str(images_root))
    _main_print("OUTPUT_ROOT =", str(output_root))
    _main_print("DETREG_PRETRAIN_CKPT =", str(pretrain_ckpt))
    _main_print("DETREG_TRAIN_FRACTIONS =", ",".join(str(x) for x in train_fractions))
    _main_print("DETREG_SEEDS =", ",".join(str(x) for x in seeds))
    _main_print("DETREG_FRACTION_SEED =", str(fraction_seed))
    _main_print("DETREG_MODEL =", detreg_model)
    _main_print("DETREG_EPOCHS =", str(epochs))
    _main_print("DETREG_BATCH_SIZE =", str(batch_size))
    _main_print("DETREG_LR_DROP =", str(lr_drop))
    _main_print("DETREG_NUM_WORKERS =", str(num_workers))
    _main_print("DETREG_DATA_LINK_MODE =", link_mode)
    _main_print("DETREG_AUTO_DOWNLOAD_PRETRAIN =", str(auto_download))
    _main_print("DETREG_EXTRA_ARGS =", " ".join(extra_args) if extra_args else "<none>")

    if not project_dir.exists():
        raise FileNotFoundError(f"PROJECT_DIR does not exist: {project_dir}")
    if not detreg_repo_dir.exists():
        raise FileNotFoundError(f"DETREG_REPO_DIR does not exist: {detreg_repo_dir}")
    if not (detreg_repo_dir / "main.py").exists():
        raise FileNotFoundError(f"DETReg main.py not found under {detreg_repo_dir}")
    if not dataset_epi_path.exists():
        raise FileNotFoundError(f"DATASET_EPI does not exist: {dataset_epi_path}")
    if not images_root.exists():
        raise FileNotFoundError(f"IMAGES_FALLBACK_ROOT does not exist: {images_root}")
    if auto_download:
        _download_if_needed(pretrain_ckpt, pretrain_url)
    if not pretrain_ckpt.exists():
        raise FileNotFoundError(f"DETREG_PRETRAIN_CKPT does not exist: {pretrain_ckpt}")
    _preflight_detreg_runtime(detreg_repo_dir=detreg_repo_dir, python_exec=python_exec)

    total_runs = len(train_fractions) * len(seeds)
    _main_print(f"EXPECTED_RUNS = {total_runs}")

    session_dir = output_root / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    epi_root = session_dir / "SquamousEpithelialCell"
    dataset_cache = session_dir / "_detreg_dataset_cache"
    resolved_cache = session_dir / "_resolved_epi_dataset"
    epi_root.mkdir(parents=True, exist_ok=True)
    dataset_cache.mkdir(parents=True, exist_ok=True)
    resolved_cache.mkdir(parents=True, exist_ok=True)

    resolved_epi = _build_resolved_dataset(dataset_epi_path, resolved_cache, images_root)

    plan_rows = []
    run_idx = 0
    for frac in train_fractions:
        for seed in seeds:
            run_idx += 1
            run_dir = epi_root / f"HPO_Config_{run_idx:03d}"
            plan_rows.append(
                {
                    "run_idx": run_idx,
                    "target": target_name,
                    "init_mode": "detreg",
                    "train_fraction": float(frac),
                    "fraction_seed": int(fraction_seed),
                    "seed": int(seed),
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "lr": (float(lr_text) if lr_text else None),
                    "lr_drop": int(lr_drop),
                    "output_dir": str(run_dir),
                }
            )
    _write_training_plan(session_dir, plan_rows)
    _main_print(f"[PLAN] Wrote plan with {len(plan_rows)} runs: {session_dir / 'TRAINING_PLAN.csv'}")

    run_records = []
    for row in plan_rows:
        run_idx = int(row["run_idx"])
        frac = float(row["train_fraction"])
        seed = int(row["seed"])
        run_dir = Path(row["output_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)

        _main_print(
            f"\n[RUN {run_idx}/{len(plan_rows)}] target={target_name} "
            f"mode=detreg frac={frac} fseed={fraction_seed} seed={seed}"
        )

        data_root = _build_detreg_coco_data_root(
            resolved_dataset_dir=resolved_epi,
            cache_root=dataset_cache,
            class_name=target_name,
            train_fraction=frac,
            fraction_seed=fraction_seed,
            link_mode=link_mode,
        )

        cmd = [
            python_exec,
            "-u",
            "main.py",
            "--output_dir",
            str(run_dir),
            "--dataset",
            "coco",
            "--dataset_file",
            "coco",
            "--data_root",
            str(data_root),
            "--pretrain",
            str(pretrain_ckpt),
            "--seed",
            str(seed),
            "--model",
            detreg_model,
            "--epochs",
            str(epochs),
            "--lr_drop",
            str(lr_drop),
            "--batch_size",
            str(batch_size),
            "--num_workers",
            str(num_workers),
            "--eval_every",
            str(eval_every),
        ]
        if lr_text:
            cmd += ["--lr", lr_text]
        cmd += extra_args

        (run_dir / "run_meta").mkdir(parents=True, exist_ok=True)
        (run_dir / "run_meta" / "train_cmd.txt").write_text(
            " ".join(shlex.quote(x) for x in cmd), encoding="utf-8"
        )
        _write_json(
            run_dir / "run_meta" / "run_config.json",
            {
                "target": target_name,
                "init_mode": "detreg",
                "train_fraction": frac,
                "fraction_seed": fraction_seed,
                "seed": seed,
                "detreg_repo_dir": str(detreg_repo_dir),
                "data_root": str(data_root),
                "pretrain_ckpt": str(pretrain_ckpt),
                "model": detreg_model,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr_drop": lr_drop,
                "lr": (float(lr_text) if lr_text else None),
                "num_workers": num_workers,
                "eval_every": eval_every,
                "extra_args": extra_args,
            },
        )

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        t0 = time.time()
        _run_and_stream(cmd, cwd=detreg_repo_dir, env=env)
        seconds = time.time() - t0

        metrics = _parse_detreg_metrics(run_dir / "log.txt")
        record = {
            "run_idx": run_idx,
            "target": target_name,
            "init_mode": "detreg",
            "train_fraction": frac,
            "fraction_seed": fraction_seed,
            "seed": seed,
            "val_AP50": metrics["val_AP50"],
            "val_mAP5095": metrics["val_mAP5095"],
            "val_AP75": metrics["val_AP75"],
            "best_epoch": metrics["best_epoch"],
            "metrics_source": metrics["metrics_source"],
            "seconds": round(seconds, 2),
            "output_dir": str(run_dir),
        }
        _write_json(run_dir / "hpo_record.json", record)
        run_records.append(record)
        _main_print(
            f"[RUN DONE] idx={run_idx} AP50={record['val_AP50']} "
            f"mAP5095={record['val_mAP5095']} best_epoch={record['best_epoch']} "
            f"seconds={record['seconds']}"
        )

    _write_json(session_dir / "RUN_SUMMARY.json", {"rows": run_records})
    _write_run_summary_csv(session_dir, run_records)
    _main_print(f"\n[SESSION DONE] {session_dir}")
    _main_print(f"[SESSION DONE] Summary: {session_dir / 'RUN_SUMMARY.csv'}")


if __name__ == "__main__":
    main()
