from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise ImportError("Pillow is required. Install with: pip install pillow") from exc


VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_STAT_DATASET_ROOT = REPO_ROOT / "SOLO_Supervised_RFDETR" / "Stat_Dataset"
DEFAULT_TARGET_TOKEN = {
    "epi": "SquamousEpithelialCell_OVR",
    "leu": "Leucocyte_OVR",
}
DEFAULT_IMAGES_ROOT = Path(r"D:\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment_Cleaned")
DEFAULT_SEMIDETR_REPO = REPO_ROOT / "_external_SemiDETR_ref"
DEFAULT_RUNS_ROOT = SCRIPT_DIR / "runs"
DEFAULT_CACHE_ROOT = SCRIPT_DIR / "cache"
DEFAULT_WSL_DISTRO = "Ubuntu-22.04"
DEFAULT_SEMIDETR_PYTHON_WSL = "/mnt/c/Users/SH37YE/AppData/Local/anaconda3/envs/semidetr/bin/python"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _norm_file_name(file_name: str) -> str:
    return file_name.replace("\\", "/").strip()


def _windows_to_wsl(path: Path) -> str:
    p = path.resolve()
    s = str(p)
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        tail = s[2:].replace("\\", "/")
        if tail.startswith("/"):
            tail = tail[1:]
        return f"/mnt/{drive}/{tail}"
    return s.replace("\\", "/")


def _auto_latest_dataset(target: str, stat_dataset_root: Path) -> Path:
    token = DEFAULT_TARGET_TOKEN[target]
    candidates = sorted([p for p in stat_dataset_root.glob(f"*{token}*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No dataset matching '*{token}*' in {stat_dataset_root}")
    return candidates[-1]


def _format_percent_tag(percent: float) -> str:
    if abs(percent - round(percent)) < 1e-8:
        return str(int(round(percent)))
    return str(percent).replace(".", "p")


def _normalize_categories(*coco_dicts: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cats = coco_dicts[0].get("categories", [])
    if not cats:
        raise ValueError("Missing categories in train annotations")
    old_ids = [int(c["id"]) for c in cats]
    id_map = {old_id: idx + 1 for idx, old_id in enumerate(sorted(old_ids))}
    norm_cats = []
    for c in cats:
        c2 = dict(c)
        c2["id"] = id_map[int(c["id"])]
        norm_cats.append(c2)

    out = []
    for js in coco_dicts:
        js2 = dict(js)
        js2["images"] = [dict(im, file_name=_norm_file_name(str(im["file_name"]))) for im in js.get("images", [])]
        js2["annotations"] = [dict(a, category_id=id_map[int(a["category_id"])]) for a in js.get("annotations", [])]
        js2["categories"] = [dict(c) for c in norm_cats]
        out.append(js2)
    return norm_cats, out


def _subset_train(train_js: dict[str, Any], labeled_percent: float, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    images = list(train_js.get("images", []))
    anns = list(train_js.get("annotations", []))
    if not images:
        raise ValueError("Train split has zero images")
    n_labeled = max(1, int(round(len(images) * labeled_percent / 100.0)))
    idx = list(range(len(images)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    keep_idx = set(idx[:n_labeled])

    sup_images = [im for i, im in enumerate(images) if i in keep_idx]
    leftover_images = [im for i, im in enumerate(images) if i not in keep_idx]
    sup_ids = {int(im["id"]) for im in sup_images}
    sup_anns = [a for a in anns if int(a["image_id"]) in sup_ids]
    sup_js = {
        "images": sup_images,
        "annotations": sup_anns,
        "categories": train_js.get("categories", []),
        "licenses": train_js.get("licenses", []),
        "info": train_js.get("info", {}),
    }
    return sup_js, leftover_images


def _scan_unlabeled_pool(images_root: Path, excluded_rel_paths: set[str], cache_root: Path) -> list[str]:
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha1(str(images_root.resolve()).encode("utf-8")).hexdigest()[:12]
    cache_file = cache_root / f"unlabeled_pool_{cache_key}.json"
    if cache_file.exists():
        payload = _read_json(cache_file)
        if payload.get("images_root") == str(images_root.resolve()):
            return [str(x) for x in payload.get("pool", [])]

    pool: list[str] = []
    for p in images_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VALID_EXTS:
            continue
        rel = p.relative_to(images_root).as_posix()
        if rel in excluded_rel_paths:
            continue
        pool.append(rel)

    _write_json(cache_file, {"images_root": str(images_root.resolve()), "pool": pool})
    return pool


def _image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as im:
        w, h = im.size
    return int(w), int(h)


def _aspect_group(width: int, height: int) -> int:
    if height <= 0:
        return 0
    return 1 if (float(width) / float(height)) > 1.0 else 0


def _aspect_group_hist(images: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[int, int] = {}
    for im in images:
        g = _aspect_group(int(im["width"]), int(im["height"]))
        counts[g] = counts.get(g, 0) + 1
    return {str(k): int(v) for k, v in sorted(counts.items())}


def _build_unsup_images(
    sup_js: dict[str, Any],
    val_js: dict[str, Any],
    test_js: dict[str, Any],
    leftover_images: list[dict[str, Any]],
    images_root: Path,
    unlabeled_per_labeled: int,
    unlabeled_max_images: int,
    unlabeled_max_width: int,
    unlabeled_max_height: int,
    seed: int,
    cache_root: Path,
) -> dict[str, Any]:
    sup_groups = {_aspect_group(int(im["width"]), int(im["height"])) for im in sup_js.get("images", [])}
    if not sup_groups:
        raise RuntimeError("Supervised split has zero images after sampling.")

    image_meta = {
        _norm_file_name(str(im["file_name"])): (int(im["width"]), int(im["height"]))
        for im in sup_js.get("images", []) + val_js.get("images", []) + test_js.get("images", []) + leftover_images
    }
    excluded = set(image_meta.keys())
    external_pool = _scan_unlabeled_pool(images_root=images_root, excluded_rel_paths=excluded, cache_root=cache_root)

    leftover_rel = [_norm_file_name(str(im["file_name"])) for im in leftover_images]
    n_sup = len(sup_js.get("images", []))
    if unlabeled_per_labeled > 0:
        target_unsup = max(1, unlabeled_per_labeled * n_sup)
    else:
        target_unsup = len(leftover_rel) + len(external_pool)
    if unlabeled_max_images > 0:
        target_unsup = min(target_unsup, unlabeled_max_images)

    rng = random.Random(seed)
    rng.shuffle(leftover_rel)
    ext = list(external_pool)
    rng.shuffle(ext)
    candidates = list(leftover_rel) + ext

    unsup_images = []
    seen: set[str] = set()
    skipped_missing = 0
    skipped_open = 0
    skipped_size = 0
    skipped_group = 0
    for rel in candidates:
        if rel in seen:
            continue
        seen.add(rel)
        wh = image_meta.get(rel)
        if wh is None:
            img_path = images_root / rel
            if not img_path.exists():
                skipped_missing += 1
                continue
            try:
                wh = _image_size(img_path)
            except Exception:
                skipped_open += 1
                continue
        if (unlabeled_max_width > 0 and wh[0] > unlabeled_max_width) or (
            unlabeled_max_height > 0 and wh[1] > unlabeled_max_height
        ):
            skipped_size += 1
            continue
        if _aspect_group(wh[0], wh[1]) not in sup_groups:
            skipped_group += 1
            continue
        unsup_images.append({"id": len(unsup_images) + 1, "file_name": rel, "width": wh[0], "height": wh[1]})
        if len(unsup_images) >= target_unsup:
            break

    if not unsup_images:
        raise RuntimeError(
            "No usable unlabeled images found. "
            f"Try increasing --unlabeled-max-width/--unlabeled-max-height or lowering --unlabeled-per-labeled. "
            f"Allowed aspect groups from labeled set: {sorted(sup_groups)}."
        )
    if len(unsup_images) < target_unsup:
        print(
            "[UNLABELED][WARN] Requested",
            target_unsup,
            "images, but prepared",
            len(unsup_images),
            (
                f"(skipped_missing={skipped_missing}, skipped_open={skipped_open}, "
                f"skipped_size={skipped_size}, skipped_group={skipped_group}, "
                f"allowed_groups={sorted(sup_groups)})."
            ),
        )

    unsup_js = {
        "images": unsup_images,
        "annotations": [],
        "categories": [dict(c) for c in sup_js.get("categories", [])],
        "licenses": sup_js.get("licenses", []),
        "info": sup_js.get("info", {}),
    }
    return unsup_js


def _write_semidetr_annotations(
    run_root: Path,
    sup_js: dict[str, Any],
    unsup_js: dict[str, Any],
    val_js: dict[str, Any],
    test_js: dict[str, Any],
    full_train_js: dict[str, Any],
    percent: float,
    seed: int,
) -> dict[str, Path]:
    ann_root = run_root / "prepared_data" / "coco" / "annotations"
    semi_root = ann_root / "semi_supervised"
    percent_tag = _format_percent_tag(percent)

    paths = {
        "sup": semi_root / f"instances_train2017.{seed}@{percent_tag}.json",
        "unsup": semi_root / f"instances_train2017.{seed}@{percent_tag}-unlabeled.json",
        "train_full": ann_root / "instances_train2017.json",
        "val": ann_root / "instances_val2017.json",
        "test": ann_root / "instances_test2017.json",
        "unlabeled_full": ann_root / "instances_unlabeled2017.json",
        "image_info_unlabeled": ann_root / "image_info_unlabeled2017.json",
    }
    _write_json(paths["sup"], sup_js)
    _write_json(paths["unsup"], unsup_js)
    _write_json(paths["train_full"], full_train_js)
    _write_json(paths["val"], val_js)
    _write_json(paths["test"], test_js)
    _write_json(paths["unlabeled_full"], unsup_js)
    _write_json(
        paths["image_info_unlabeled"],
        {
            "images": unsup_js.get("images", []),
            "annotations": [],
            "categories": unsup_js.get("categories", []),
            "licenses": unsup_js.get("licenses", []),
            "info": unsup_js.get("info", {}),
        },
    )
    return paths


def _write_run_config(
    run_root: Path,
    semidetr_repo: Path,
    ann_paths: dict[str, Path],
    images_root: Path,
    class_name: str,
    sample_ratio: list[int],
    pseudo_thr: float,
    unsup_weight: float,
    max_iters: int,
    eval_interval: int,
    ckpt_interval: int,
    samples_per_gpu: int,
    workers_per_gpu: int,
    dist_backend: str,
    use_tensorboard: bool,
    load_from: Path | None,
    backbone_checkpoint: Path | None,
) -> Path:
    cfg_path = run_root / "generated_config.py"
    repo_wsl = _windows_to_wsl(semidetr_repo)
    work_dir_wsl = _windows_to_wsl(run_root / "work_dir")
    img_root_wsl = _windows_to_wsl(images_root)
    ann_sup_wsl = _windows_to_wsl(ann_paths["sup"])
    ann_unsup_wsl = _windows_to_wsl(ann_paths["unsup"])
    ann_val_wsl = _windows_to_wsl(ann_paths["val"])
    ann_test_wsl = _windows_to_wsl(ann_paths["test"])
    load_from_line = f"load_from = r'''{_windows_to_wsl(load_from)}'''\n" if load_from else ""
    backbone_line = (
        f"model = dict(backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=r'''{_windows_to_wsl(backbone_checkpoint)}''')), bbox_head=dict(num_classes=1))\n\n"
        if backbone_checkpoint
        else "model = dict(backbone=dict(init_cfg=None), bbox_head=dict(num_classes=1))\n\n"
    )

    log_hooks = "[dict(type='TextLoggerHook')]"
    if use_tensorboard:
        log_hooks = "[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')]"

    cfg_text = (
        f"_base_ = r'''{repo_wsl}/configs/detr_ssod/base_dino_detr_ssod_coco.py'''\n\n"
        f"{backbone_line}"
        "data = dict(\n"
        f"    samples_per_gpu={samples_per_gpu},\n"
        f"    workers_per_gpu={workers_per_gpu},\n"
        "    train=dict(\n"
        f"        sup=dict(type='CocoDataset', ann_file=r'''{ann_sup_wsl}''', img_prefix=r'''{img_root_wsl}/''', classes=({class_name!r},)),\n"
        f"        unsup=dict(type='CocoDataset', ann_file=r'''{ann_unsup_wsl}''', img_prefix=r'''{img_root_wsl}/''', classes=({class_name!r},), filter_empty_gt=False),\n"
        "    ),\n"
        f"    val=dict(type='CocoDataset', ann_file=r'''{ann_val_wsl}''', img_prefix=r'''{img_root_wsl}/''', classes=({class_name!r},)),\n"
        f"    test=dict(type='CocoDataset', ann_file=r'''{ann_test_wsl}''', img_prefix=r'''{img_root_wsl}/''', classes=({class_name!r},)),\n"
        f"    sampler=dict(train=dict(sample_ratio={sample_ratio})),\n"
        ")\n\n"
        "semi_wrapper = dict(\n"
        "    type='DinoDetrSSOD',\n"
        "    model='${model}',\n"
        f"    train_cfg=dict(use_teacher_proposal=False, pseudo_label_initial_score_thr={pseudo_thr}, min_pseduo_box_size=0, unsup_weight={unsup_weight}),\n"
        "    test_cfg=dict(inference_on='student'),\n"
        ")\n\n"
        "custom_hooks = [\n"
        "    dict(type='NumClassCheckHook'),\n"
        "    dict(type='MeanTeacher', momentum=0.999, interval=1, warm_up=0),\n"
        "    dict(type='StepRecord', normalize=False),\n"
        "]\n\n"
        f"evaluation = dict(type='SubModulesDistEvalHook', interval={eval_interval})\n"
        f"runner = dict(_delete_=True, type='IterBasedRunner', max_iters={max_iters})\n"
        f"checkpoint_config = dict(by_epoch=False, interval={ckpt_interval}, max_keep_ckpts=5, create_symlink=False)\n"
        f"log_config = dict(interval=50, hooks={log_hooks})\n"
        f"dist_params = dict(backend={dist_backend!r})\n"
        f"{load_from_line}"
        f"work_dir = r'''{work_dir_wsl}'''\n"
    )
    cfg_path.write_text(cfg_text, encoding="utf-8")
    return cfg_path


def _verify_semidetr_runtime(args: argparse.Namespace, semidetr_repo: Path) -> None:
    repo_wsl = _windows_to_wsl(semidetr_repo)
    env_lib = "/mnt/c/Users/SH37YE/AppData/Local/anaconda3/envs/semidetr/lib"
    torch_lib = "/mnt/c/Users/SH37YE/AppData/Local/anaconda3/envs/semidetr/lib/python3.8/site-packages/torch/lib"
    cuda_runtime_lib = "/mnt/c/Users/SH37YE/AppData/Local/anaconda3/envs/semidetr/lib/python3.8/site-packages/nvidia/cuda_runtime/lib"
    ld = f"{torch_lib}:{cuda_runtime_lib}:{env_lib}"
    script = (
        f"export LD_LIBRARY_PATH={shlex.quote(ld)}; "
        f"export PYTHONPATH={shlex.quote(repo_wsl)}:{shlex.quote(repo_wsl + '/thirdparty/mmdetection')}:${{PYTHONPATH:-}}; "
        f"{shlex.quote(args.semidetr_python_wsl)} -c 'import torch; import mmdet; import detr_ssod; import MultiScaleDeformableAttention; print(torch.__version__, torch.cuda.is_available())'"
    )
    cmd = ["wsl", "-d", args.wsl_distro, "--", "bash", "-lc", script]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(
            "Semi-DETR runtime check failed.\n"
            f"STDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}\n"
            "Install mmdetection + detr_ssod in semidetr env and set LD_LIBRARY_PATH for torch/cuda runtime."
        )


def _pick_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _launch_training(args: argparse.Namespace, cfg_path: Path, semidetr_repo: Path, run_root: Path) -> None:
    repo_wsl = _windows_to_wsl(semidetr_repo)
    cfg_wsl = _windows_to_wsl(cfg_path)
    work_dir_wsl = _windows_to_wsl(run_root / "work_dir")
    env_lib = "/mnt/c/Users/SH37YE/AppData/Local/anaconda3/envs/semidetr/lib"
    torch_lib = "/mnt/c/Users/SH37YE/AppData/Local/anaconda3/envs/semidetr/lib/python3.8/site-packages/torch/lib"
    cuda_runtime_lib = "/mnt/c/Users/SH37YE/AppData/Local/anaconda3/envs/semidetr/lib/python3.8/site-packages/nvidia/cuda_runtime/lib"
    ld = f"{torch_lib}:{cuda_runtime_lib}:{env_lib}"

    launch_mode = str(args.launcher).strip().lower()
    if launch_mode == "pytorch":
        if int(args.samples_per_gpu) < 2:
            raise ValueError("For launcher=pytorch, samples-per-gpu must be >= 2 (SemiBalanceSampler requirement).")
        if int(args.master_port) > 0:
            master_port = int(args.master_port)
        else:
            master_port = _pick_free_tcp_port()
        train_step = (
            f"{shlex.quote(args.semidetr_python_wsl)} -m torch.distributed.launch "
            f"--nproc_per_node=1 --master_port={master_port} "
            f"tools/train_detr_ssod.py {shlex.quote(cfg_wsl)} "
            f"--work-dir {shlex.quote(work_dir_wsl)} --launcher pytorch --seed {int(args.seed)}"
        )
    elif launch_mode == "none":
        train_step = (
            f"{shlex.quote(args.semidetr_python_wsl)} tools/train_detr_ssod.py {shlex.quote(cfg_wsl)} "
            f"--work-dir {shlex.quote(work_dir_wsl)} --launcher none --seed {int(args.seed)}"
        )
    else:
        raise ValueError(f"Unsupported launcher mode: {args.launcher}")

    train_cmd = (
        f"export LD_LIBRARY_PATH={shlex.quote(ld)}; "
        f"export PYTHONPATH={shlex.quote(repo_wsl)}:{shlex.quote(repo_wsl + '/thirdparty/mmdetection')}:${{PYTHONPATH:-}}; "
        f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(args.cuda_visible_devices))}; "
        f"cd {shlex.quote(repo_wsl)}; "
        f"{train_step}"
    )
    cmd = ["wsl", "-d", args.wsl_distro, "--", "bash", "-lc", train_cmd]
    print("\n[RUN] launching command:")
    print(" ".join(shlex.quote(x) for x in cmd))
    if args.dry_run:
        print("[DRY RUN] skipped launch.")
        return
    subprocess.run(cmd, check=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare and launch Semi-DETR locally (Windows + WSL).")
    p.add_argument("--target", choices=["epi", "leu"], default="epi")
    p.add_argument("--dataset-dir", type=str, default="")
    p.add_argument("--stat-dataset-root", type=str, default=str(DEFAULT_STAT_DATASET_ROOT))
    p.add_argument("--images-root", type=str, default=str(DEFAULT_IMAGES_ROOT))
    p.add_argument("--semidetr-repo", type=str, default=str(DEFAULT_SEMIDETR_REPO))
    p.add_argument("--runs-root", type=str, default=str(DEFAULT_RUNS_ROOT))
    p.add_argument("--cache-root", type=str, default=str(DEFAULT_CACHE_ROOT))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--labeled-percent", type=float, default=50.0)
    p.add_argument("--unlabeled-per-labeled", type=int, default=6)
    p.add_argument("--unlabeled-max-images", type=int, default=0)
    p.add_argument("--unlabeled-max-width", type=int, default=2048)
    p.add_argument("--unlabeled-max-height", type=int, default=2048)
    p.add_argument("--samples-per-gpu", type=int, default=5)
    p.add_argument("--workers-per-gpu", type=int, default=5)
    p.add_argument("--sample-ratio", type=str, default="1,4")
    p.add_argument("--pseudo-thresh", type=float, default=0.5)
    p.add_argument("--unsup-weight", type=float, default=4.0)
    p.add_argument("--max-iters", type=int, default=120000)
    p.add_argument("--eval-interval", type=int, default=4000)
    p.add_argument("--ckpt-interval", type=int, default=4000)
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--load-from", type=str, default="")
    p.add_argument("--backbone-checkpoint", type=str, default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-env-check", action="store_true")
    p.add_argument("--wsl-distro", type=str, default=DEFAULT_WSL_DISTRO)
    p.add_argument("--semidetr-python-wsl", type=str, default=DEFAULT_SEMIDETR_PYTHON_WSL)
    p.add_argument("--cuda-visible-devices", type=str, default="0")
    p.add_argument("--launcher", choices=["pytorch", "none"], default="pytorch")
    p.add_argument("--dist-backend", choices=["gloo", "nccl"], default="gloo")
    p.add_argument("--use-tensorboard", action="store_true")
    p.add_argument("--master-port", type=int, default=0)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    stat_dataset_root = Path(args.stat_dataset_root).expanduser().resolve()
    images_root = Path(args.images_root).expanduser().resolve()
    semidetr_repo = Path(args.semidetr_repo).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    dataset_dir = (
        Path(args.dataset_dir).expanduser().resolve()
        if args.dataset_dir
        else _auto_latest_dataset(args.target, stat_dataset_root)
    )
    load_from = Path(args.load_from).expanduser().resolve() if args.load_from else None
    backbone_checkpoint = Path(args.backbone_checkpoint).expanduser().resolve() if args.backbone_checkpoint else None

    if not images_root.exists():
        raise FileNotFoundError(f"images-root does not exist: {images_root}")
    if not semidetr_repo.exists():
        raise FileNotFoundError(f"semidetr-repo does not exist: {semidetr_repo}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset-dir does not exist: {dataset_dir}")

    run_tag = args.run_name.strip() or f"{time.strftime('%Y%m%d-%H%M%S')}_{args.target}_p{_format_percent_tag(args.labeled_percent)}_s{args.seed}"
    run_root = runs_root / run_tag
    run_root.mkdir(parents=True, exist_ok=True)

    train_js = _read_json(dataset_dir / "train" / "_annotations.coco.json")
    val_js = _read_json(dataset_dir / "valid" / "_annotations.coco.json")
    test_js = _read_json(dataset_dir / "test" / "_annotations.coco.json")
    _, (train_js, val_js, test_js) = _normalize_categories(train_js, val_js, test_js)

    for js in (train_js, val_js, test_js):
        for im in js.get("images", []):
            im["file_name"] = _norm_file_name(str(im["file_name"]))

    sup_js, leftover_images = _subset_train(train_js, labeled_percent=float(args.labeled_percent), seed=int(args.seed))
    unsup_js = _build_unsup_images(
        sup_js=sup_js,
        val_js=val_js,
        test_js=test_js,
        leftover_images=leftover_images,
        images_root=images_root,
        unlabeled_per_labeled=int(args.unlabeled_per_labeled),
        unlabeled_max_images=int(args.unlabeled_max_images),
        unlabeled_max_width=int(args.unlabeled_max_width),
        unlabeled_max_height=int(args.unlabeled_max_height),
        seed=int(args.seed),
        cache_root=cache_root,
    )

    ann_paths = _write_semidetr_annotations(
        run_root=run_root,
        sup_js=sup_js,
        unsup_js=unsup_js,
        val_js=val_js,
        test_js=test_js,
        full_train_js=train_js,
        percent=float(args.labeled_percent),
        seed=int(args.seed),
    )

    class_name = str((sup_js.get("categories") or [{"name": "object"}])[0]["name"])
    sample_ratio = [int(x.strip()) for x in str(args.sample_ratio).split(",") if x.strip()]
    if len(sample_ratio) != 2:
        raise ValueError("--sample-ratio must be two integers, e.g. 1,4")

    cfg_path = _write_run_config(
        run_root=run_root,
        semidetr_repo=semidetr_repo,
        ann_paths=ann_paths,
        images_root=images_root,
        class_name=class_name,
        sample_ratio=sample_ratio,
        pseudo_thr=float(args.pseudo_thresh),
        unsup_weight=float(args.unsup_weight),
        max_iters=int(args.max_iters),
        eval_interval=int(args.eval_interval),
        ckpt_interval=int(args.ckpt_interval),
        samples_per_gpu=int(args.samples_per_gpu),
        workers_per_gpu=int(args.workers_per_gpu),
        dist_backend=str(args.dist_backend),
        use_tensorboard=bool(args.use_tensorboard),
        load_from=load_from,
        backbone_checkpoint=backbone_checkpoint,
    )

    summary = {
        "run_root": str(run_root),
        "dataset_dir": str(dataset_dir),
        "images_root": str(images_root),
        "target": args.target,
        "labeled_percent": float(args.labeled_percent),
        "seed": int(args.seed),
        "labeled_images": len(sup_js.get("images", [])),
        "labeled_annotations": len(sup_js.get("annotations", [])),
        "labeled_aspect_groups": _aspect_group_hist(sup_js.get("images", [])),
        "unlabeled_images": len(unsup_js.get("images", [])),
        "unlabeled_aspect_groups": _aspect_group_hist(unsup_js.get("images", [])),
        "unlabeled_max_width": int(args.unlabeled_max_width),
        "unlabeled_max_height": int(args.unlabeled_max_height),
        "dist_backend": str(args.dist_backend),
        "use_tensorboard": bool(args.use_tensorboard),
        "config_path": str(cfg_path),
    }
    _write_json(run_root / "run_summary.json", summary)
    print("[PREP] run summary:")
    print(json.dumps(summary, indent=2))

    if not args.skip_env_check:
        _verify_semidetr_runtime(args=args, semidetr_repo=semidetr_repo)
        print("[CHECK] runtime import check passed.")

    _launch_training(args=args, cfg_path=cfg_path, semidetr_repo=semidetr_repo, run_root=run_root)


if __name__ == "__main__":
    main()
