from __future__ import annotations

import csv
import json
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
UCLOUD_SEMIDETR_PY = Path("/work/CondaEnv/miniconda3/envs/semidetr/bin/python")


def _csv_tokens(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _csv_float(raw: str) -> list[float]:
    vals = [float(x) for x in _csv_tokens(raw)]
    if not vals:
        raise ValueError("Expected non-empty float CSV")
    return vals


def _csv_int(raw: str) -> list[int]:
    vals = [int(x) for x in _csv_tokens(raw)]
    if not vals:
        raise ValueError("Expected non-empty int CSV")
    return vals


def _parse_bool(raw: str) -> bool:
    x = str(raw).strip().lower()
    if x in {"1", "true", "yes", "y", "on"}:
        return True
    if x in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Expected bool text, got: {raw!r}")


def _short_frac_tag(frac: float) -> str:
    return str(f"{float(frac):.5f}").rstrip("0").rstrip(".").replace(".", "p")


def _auto_latest_dataset(dataset_root: Path, token: str) -> str:
    cands = sorted([p for p in dataset_root.glob(f"*{token}*") if p.is_dir()])
    return str(cands[-1]) if cands else ""


def _detect_runtime_profile() -> str:
    forced = os.getenv("RFDETR_RUNTIME_PROFILE", "auto").strip().lower()
    if forced in {"ucloud", "local"}:
        return forced
    if forced not in {"", "auto"}:
        raise ValueError("RFDETR_RUNTIME_PROFILE must be one of: auto, ucloud, local")
    if os.name != "nt" and Path("/work").exists():
        has_member = any(Path("/work").glob("Member Files:*"))
        has_hash = any(Path("/work").glob("*#*"))
        if has_member or has_hash:
            return "ucloud"
    return "local"


def _auto_prepare_semidetr_env(target_py: Path) -> None:
    runtime = _detect_runtime_profile()
    if runtime != "ucloud":
        return
    if target_py.exists():
        return
    if not _parse_bool(os.getenv("SEMIDETR_AUTO_PREPARE_ENV", "1")):
        return
    print(f"[BOOT] Preparing Semi-DETR env because interpreter is missing: {target_py}")
    prep = r"""
set -Eeuo pipefail
eval "$(/work/CondaEnv/miniconda3/bin/conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx semidetr; then
  conda create -y -n semidetr python=3.8
fi
conda activate semidetr
python -m pip install -U pip setuptools wheel openmim
if python - <<'PY'
import importlib.util
mods = ("torch", "mmcv")
missing = [m for m in mods if importlib.util.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
then
  :
else
  python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
  python -m mim install "mmcv-full>=1.3.8,<=1.4.0"
fi
"""
    p = subprocess.run(["bash", "-lc", prep], capture_output=True, text=True)
    if p.returncode != 0:
        msg = (p.stdout or "") + "\n" + (p.stderr or "")
        raise RuntimeError(
            "Failed to auto-prepare semidetr environment.\n"
            f"Expected interpreter: {target_py}\n"
            f"Details:\n{msg.strip()}"
        )
    if not target_py.exists():
        raise FileNotFoundError(f"Semi-DETR env prep completed but interpreter still missing: {target_py}")


def _maybe_reexec_with_semidetr_python() -> None:
    # Guard against recursively re-execing ourselves.
    if os.getenv("_SEMIDETR_REEXEC_DONE", "") == "1":
        return
    runtime = _detect_runtime_profile()
    preferred_txt = os.getenv("SEMIDETR_PYTHON", "").strip()
    preferred = Path(preferred_txt).expanduser() if preferred_txt else None
    if preferred is None and runtime == "ucloud" and UCLOUD_SEMIDETR_PY.exists():
        preferred = UCLOUD_SEMIDETR_PY
        os.environ["SEMIDETR_PYTHON"] = str(preferred)
    if preferred is None:
        return
    if not preferred.exists():
        _auto_prepare_semidetr_env(preferred)
    if not preferred.exists():
        raise FileNotFoundError(
            f"SEMIDETR_PYTHON points to a missing interpreter: {preferred}. "
            "Prepare the semidetr env and set SEMIDETR_PYTHON correctly."
        )
    curr = Path(sys.executable).resolve()
    want = preferred.resolve()
    if curr == want:
        return
    print(f"[BOOT] Re-exec with Semi-DETR interpreter: {want} (current={curr})")
    env = os.environ.copy()
    env["_SEMIDETR_REEXEC_DONE"] = "1"
    os.execve(str(want), [str(want)] + sys.argv, env)

_maybe_reexec_with_semidetr_python()
RUNTIME_PROFILE = _detect_runtime_profile()
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DATASET_ROOT_CANDIDATES = [
    REPO_ROOT / "SOLO_Supervised_RFDETR" / "Stat_Dataset",
    REPO_ROOT / "Stat_Dataset",
]
DATASET_ROOT_DEFAULT = next((p for p in DATASET_ROOT_CANDIDATES if p.exists()), DATASET_ROOT_CANDIDATES[0])

if not os.getenv("DATASET_EPI", "").strip():
    v = _auto_latest_dataset(DATASET_ROOT_DEFAULT, "SquamousEpithelialCell_OVR")
    if v:
        os.environ["DATASET_EPI"] = v
if not os.getenv("DATASET_LEUCO", "").strip():
    v = _auto_latest_dataset(DATASET_ROOT_DEFAULT, "Leucocyte_OVR")
    if v:
        os.environ["DATASET_LEUCO"] = v

TARGET = os.getenv("SEMIDETR_TARGET", "epi").strip().lower()
if TARGET not in {"epi", "leu"}:
    raise ValueError("SEMIDETR_TARGET must be one of: epi, leu")

INIT_MODE = os.getenv("SEMIDETR_INIT_MODE", "scratch").strip().lower()
if INIT_MODE not in {"default", "scratch"}:
    raise ValueError("SEMIDETR_INIT_MODE must be one of: default, scratch")

FRACTIONS = _csv_float(os.getenv("SEMIDETR_TRAIN_FRACTIONS", "1.0,0.5"))
SEEDS = _csv_int(os.getenv("SEMIDETR_SEEDS", "42"))
FRACTION_SEEDS = _csv_int(os.getenv("SEMIDETR_FRACTION_SEEDS", "42"))
if len(FRACTION_SEEDS) == 1 and len(SEEDS) > 1:
    FRACTION_SEEDS = [FRACTION_SEEDS[0] for _ in SEEDS]
if len(FRACTION_SEEDS) not in {1, len(SEEDS)}:
    raise ValueError("SEMIDETR_FRACTION_SEEDS must have length 1 or len(SEMIDETR_SEEDS)")

SEMI_DETR_EXPECTED_CFG_REL = Path("configs") / "detr_ssod" / "detr_ssod_dino_detr_r50_coco_120k.py"
SEMI_DETR_REPO = Path(os.getenv("SEMIDETR_REPO_DIR", str(REPO_ROOT / "_external_SemiDETR_ref"))).expanduser()
SEMI_DETR_TRAIN_SCRIPT = SEMI_DETR_REPO / "tools" / "train_detr_ssod.py"
SEMI_DETR_BASE_CFG = Path(
    os.getenv(
        "SEMIDETR_BASE_CONFIG",
        str(SEMI_DETR_REPO / SEMI_DETR_EXPECTED_CFG_REL),
    )
).expanduser()

OUTPUT_ROOT = Path(
    os.getenv(
        "SEMIDETR_OUTPUT_ROOT",
        str((REPO_ROOT / "SOLO_SemiSupervised_RFDETR" / "SemiDETR_OUTPUT_local")
            if RUNTIME_PROFILE == "local"
            else (Path("/work") / os.getenv("USER_BASE_DIR", "") / "SemiDETR_OUTPUT")),
    )
).expanduser().resolve()
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

if RUNTIME_PROFILE == "local":
    IMAGES_FALLBACK_DEFAULT = REPO_ROOT / "CellScanData" / "Zoom10x - Quality Assessment_Cleaned"
else:
    IMAGES_FALLBACK_DEFAULT = Path("/work") / os.getenv("USER_BASE_DIR", "") / "CellScanData" / "Zoom10x - Quality Assessment_Cleaned"
IMAGES_FALLBACK_ROOT = Path(os.getenv("IMAGES_FALLBACK_ROOT", str(IMAGES_FALLBACK_DEFAULT))).expanduser().resolve()

NUM_CLASSES = 1
MODEL_CLASS_NAME = os.getenv("SEMIDETR_MODEL", "DINO-R50").strip()
MAX_ITERS = int(os.getenv("SEMIDETR_MAX_ITERS", "120000"))
EVAL_INTERVAL = int(os.getenv("SEMIDETR_EVAL_INTERVAL", "4000"))
CKPT_INTERVAL = int(os.getenv("SEMIDETR_CKPT_INTERVAL", "4000"))
SAMPLES_PER_GPU = int(os.getenv("SEMIDETR_SAMPLES_PER_GPU", "5"))
WORKERS_PER_GPU = int(os.getenv("SEMIDETR_WORKERS_PER_GPU", "5"))
UNSUP_RATIO = os.getenv("SEMIDETR_SAMPLE_RATIO", "1,4").strip()
PSEUDO_THRESH = float(os.getenv("SEMIDETR_PSEUDO_SCORE_THRESH", "0.5"))
UNSUP_WEIGHT = float(os.getenv("SEMIDETR_UNSUP_WEIGHT", "4.0"))
UNLABELED_MAX_IMAGES = int(os.getenv("SEMIDETR_UNLABELED_MAX_IMAGES", "0"))
UNLABELED_IMAGES_PER_LABELED = int(os.getenv("SEMIDETR_UNLABELED_IMAGES_PER_LABELED", "6"))
CONTINUE_ON_ERROR = _parse_bool(os.getenv("SEMIDETR_CONTINUE_ON_ERROR", "0"))
DRY_RUN = _parse_bool(os.getenv("SEMIDETR_DRY_RUN", "0"))
BOOTSTRAP_DEPS = _parse_bool(os.getenv("SEMIDETR_BOOTSTRAP_DEPS", "0"))


def _require_exists(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")


def _semidetr_repo_is_valid(repo_dir: Path) -> bool:
    return (
        repo_dir.exists()
        and (repo_dir / "tools" / "train_detr_ssod.py").exists()
        and (repo_dir / SEMI_DETR_EXPECTED_CFG_REL).exists()
    )


def _resolve_semidetr_repo_and_cfg() -> tuple[Path, Path, Path]:
    repo_env = os.getenv("SEMIDETR_REPO_DIR", "").strip()
    candidates: list[Path] = []
    if repo_env:
        candidates.append(Path(repo_env).expanduser())
    candidates.extend(
        [
            REPO_ROOT / "_external_SemiDETR_ref",
            REPO_ROOT / "_external_SemiDETR_ref_runtime",
            SCRIPT_DIR / "_external_SemiDETR_ref",
            Path("/work/projects/myproj/_external_SemiDETR_ref"),
            Path("/work/projects/myproj/_external_SemiDETR_ref_runtime"),
        ]
    )
    for root in [REPO_ROOT, Path("/work/projects/myproj")]:
        if root.exists():
            for p in sorted(root.glob("_external_SemiDETR_ref_runtime_*"), reverse=True):
                candidates.append(p)

    repo = next((p for p in candidates if _semidetr_repo_is_valid(p)), None)
    if repo is None:
        auto_clone = _parse_bool(os.getenv("SEMIDETR_AUTO_CLONE", "1"))
        if not auto_clone:
            raise FileNotFoundError(
                "Could not find a valid Semi-DETR repo (missing tools/train_detr_ssod.py "
                f"or {SEMI_DETR_EXPECTED_CFG_REL}). "
                "Set SEMIDETR_REPO_DIR to a full clone or set SEMIDETR_AUTO_CLONE=1."
            )

        clone_target_raw = os.getenv("SEMIDETR_CLONE_TARGET", "").strip()
        clone_target = Path(clone_target_raw).expanduser() if clone_target_raw else (REPO_ROOT / "_external_SemiDETR_ref_runtime")
        if clone_target.exists() and not _semidetr_repo_is_valid(clone_target):
            if any(clone_target.iterdir()):
                clone_target = REPO_ROOT / f"_external_SemiDETR_ref_runtime_{int(time.time())}"
            else:
                clone_target.rmdir()

        def _try_clone(dst: Path) -> tuple[bool, str]:
            cp = subprocess.run(
                ["git", "clone", "https://github.com/JCZ404/Semi-DETR", str(dst)],
                capture_output=True,
                text=True,
            )
            details = ((cp.stdout or "") + "\n" + (cp.stderr or "")).strip()
            return cp.returncode == 0, details

        print(f"[SETUP] Cloning official Semi-DETR repo to {clone_target}")
        ok, details = _try_clone(clone_target)
        if not ok and _semidetr_repo_is_valid(clone_target):
            ok = True
        if not ok:
            alt = REPO_ROOT / f"_external_SemiDETR_ref_runtime_{int(time.time())}"
            print(f"[SETUP] First clone failed, retrying at {alt}")
            ok2, details2 = _try_clone(alt)
            if ok2:
                clone_target = alt
                ok = True
                details = details2
            elif _semidetr_repo_is_valid(alt):
                clone_target = alt
                ok = True
                details = details2
            else:
                raise RuntimeError(
                    "Failed to clone Semi-DETR repo automatically.\n"
                    f"First attempt details:\n{details}\n\n"
                    f"Second attempt details:\n{details2}"
                )
        repo = clone_target

    if not _semidetr_repo_is_valid(repo):
        raise FileNotFoundError(
            f"Resolved SEMIDETR_REPO_DIR is invalid: {repo} "
            f"(missing {SEMI_DETR_EXPECTED_CFG_REL} or tools/train_detr_ssod.py)"
        )

    cfg_env = os.getenv("SEMIDETR_BASE_CONFIG", "").strip()
    if cfg_env:
        cfg = Path(cfg_env).expanduser()
        if not cfg.exists():
            fallback = repo / SEMI_DETR_EXPECTED_CFG_REL
            if fallback.exists():
                cfg = fallback
            else:
                raise FileNotFoundError(f"Configured SEMIDETR_BASE_CONFIG does not exist: {cfg_env}")
    else:
        cfg = repo / SEMI_DETR_EXPECTED_CFG_REL

    return repo.resolve(), (repo / "tools" / "train_detr_ssod.py").resolve(), cfg.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _resolve_dataset_dir() -> tuple[Path, str]:
    if TARGET == "epi":
        p = Path(os.getenv("DATASET_EPI", "")).expanduser().resolve()
        return p, "Squamous Epithelial Cell"
    p = Path(os.getenv("DATASET_LEUCO", "")).expanduser().resolve()
    return p, "Leucocyte"


def _index_images(root: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_rel: dict[str, Path] = {}
    by_name: dict[str, list[Path]] = {}
    root = root.resolve()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            rel = p.resolve().relative_to(root).as_posix()
            by_rel[rel] = p
            by_name.setdefault(p.name, []).append(p)
    return by_rel, by_name


def _resolve_image_path(file_name: str, images_root: Path, by_rel: dict[str, Path], by_name: dict[str, list[Path]]) -> Path:
    rel = file_name.replace("\\", "/")
    direct = images_root / rel
    if direct.exists():
        return direct.resolve()
    if rel in by_rel:
        return by_rel[rel]
    base = Path(rel).name
    if base in by_name:
        cands = sorted(by_name[base], key=lambda q: len(str(q)))
        return cands[0].resolve()
    try_abs = Path(file_name)
    if try_abs.is_absolute() and try_abs.exists():
        return try_abs.resolve()
    raise FileNotFoundError(f"Could not resolve image '{file_name}' under {images_root}")


def build_resolved_static_dataset(src_dir: Path, dst_dir: Path, images_root: Path) -> Path:
    ok = dst_dir / ".RESOLVED_OK"
    if ok.exists():
        print(f"[RESOLVE] Using cached dataset: {dst_dir}")
        return dst_dir

    _require_exists(images_root, "IMAGES_FALLBACK_ROOT")
    by_rel, by_name = _index_images(images_root)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        src_json = src_dir / split / "_annotations.coco.json"
        if not src_json.exists():
            continue
        data = _load_json(src_json)
        out_images = []
        for im in data.get("images", []):
            try:
                rp = _resolve_image_path(str(im["file_name"]), images_root, by_rel, by_name)
            except FileNotFoundError as e:
                print(f"[RESOLVE][WARN] {e}")
                continue
            im2 = dict(im)
            im2["file_name"] = str(rp)
            out_images.append(im2)
        valid_ids = {im["id"] for im in out_images}
        out_anns = [a for a in data.get("annotations", []) if a.get("image_id") in valid_ids]
        out = {
            "images": out_images,
            "annotations": out_anns,
            "categories": data.get("categories", []),
        }
        _write_json(dst_dir / split / "_annotations.coco.json", out)
        print(f"[RESOLVE] {split}: images={len(out_images)} anns={len(out_anns)}")

    ok.write_text("ok", encoding="utf-8")
    return dst_dir


def build_fractional_train_split(resolved_root: Path, frac: float, cache_root: Path, seed: int = 42) -> Path:
    if frac >= 0.999:
        return resolved_root
    tag = f"trainfrac_{_short_frac_tag(frac)}_seed{int(seed)}"
    dst = cache_root / f"{resolved_root.name}_{tag}"
    ok = dst / ".FRACTION_OK"
    if ok.exists():
        print(f"[FRACTION] Using cached split: {dst}")
        return dst

    for split in ("valid", "test"):
        src = resolved_root / split / "_annotations.coco.json"
        if src.exists():
            _write_json(dst / split / "_annotations.coco.json", _load_json(src))

    src_train = resolved_root / "train" / "_annotations.coco.json"
    data = _load_json(src_train)
    images = list(data.get("images", []))
    anns = list(data.get("annotations", []))
    cats = list(data.get("categories", []))
    idx = list(range(len(images)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_keep = max(1, int(round(len(images) * frac)))
    keep_idx = set(idx[:n_keep])
    kept_images = [im for i, im in enumerate(images) if i in keep_idx]
    kept_ids = {im["id"] for im in kept_images}
    kept_anns = [a for a in anns if a.get("image_id") in kept_ids]
    _write_json(
        dst / "train" / "_annotations.coco.json",
        {"images": kept_images, "annotations": kept_anns, "categories": cats},
    )
    ok.write_text("ok", encoding="utf-8")
    print(f"[FRACTION] train: kept {len(kept_images)}/{len(images)} images")
    return dst


def _collect_resolved_paths(dataset_root: Path) -> set[str]:
    out: set[str] = set()
    for split in ("train", "valid", "test"):
        p = dataset_root / split / "_annotations.coco.json"
        if not p.exists():
            continue
        for im in _load_json(p).get("images", []):
            fn = str(im.get("file_name", "")).strip()
            if fn:
                out.add(str(Path(fn).resolve()))
    return out


def _build_unlabeled_pool(images_root: Path, excluded: set[str], cache_json: Path) -> list[str]:
    if cache_json.exists():
        try:
            js = _load_json(cache_json)
            items = js.get("paths", [])
            if isinstance(items, list):
                return [str(x) for x in items]
        except Exception:
            pass
    paths: list[str] = []
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            rp = str(p.resolve())
            if rp not in excluded:
                paths.append(rp)
    paths.sort()
    _write_json(cache_json, {"paths": paths})
    return paths


def _make_semidetr_ann_files(
    labeled_dataset: Path,
    run_prep_dir: Path,
    unlabeled_pool: list[str],
    categories: list[dict[str, Any]],
    seed: int,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    sup_train = _load_json(labeled_dataset / "train" / "_annotations.coco.json")
    val_js = _load_json(labeled_dataset / "valid" / "_annotations.coco.json")
    test_js = _load_json(labeled_dataset / "test" / "_annotations.coco.json")

    labeled_n = len(sup_train.get("images", []))
    if UNLABELED_IMAGES_PER_LABELED > 0:
        n_unsup = max(1, UNLABELED_IMAGES_PER_LABELED * labeled_n)
    else:
        n_unsup = len(unlabeled_pool)
    if UNLABELED_MAX_IMAGES > 0:
        n_unsup = min(n_unsup, UNLABELED_MAX_IMAGES)
    n_unsup = min(n_unsup, len(unlabeled_pool))

    picks = list(unlabeled_pool)
    random.Random(seed).shuffle(picks)
    picks = picks[:n_unsup]

    unsup_images = []
    for i, fp in enumerate(picks, start=1):
        try:
            from PIL import Image

            with Image.open(fp) as im:
                w, h = im.size
        except Exception:
            w, h = 0, 0
        unsup_images.append(
            {
                "id": i,
                "file_name": str(Path(fp).resolve()),
                "width": int(w),
                "height": int(h),
            }
        )
    unsup_js = {"images": unsup_images, "annotations": [], "categories": categories}

    ann_sup = run_prep_dir / "instances_train_labeled.json"
    ann_unsup = run_prep_dir / "instances_train_unlabeled.json"
    ann_val = run_prep_dir / "instances_val.json"
    ann_test = run_prep_dir / "instances_test.json"

    _write_json(ann_sup, sup_train)
    _write_json(ann_unsup, unsup_js)
    _write_json(ann_val, val_js)
    _write_json(ann_test, test_js)

    report = {
        "labeled_images": labeled_n,
        "labeled_annotations": len(sup_train.get("annotations", [])),
        "candidate_unlabeled_images": len(unlabeled_pool),
        "selected_unlabeled_images": len(unsup_images),
    }
    _write_json(run_prep_dir / "dataset_report.json", report)
    return ann_sup, ann_unsup, ann_val, ann_test, report


def _make_run_cfg(
    cfg_path: Path,
    ann_sup: Path,
    ann_unsup: Path,
    ann_val: Path,
    ann_test: Path,
    class_name: str,
    work_dir: Path,
) -> None:
    ratio_tokens = [int(x) for x in _csv_tokens(UNSUP_RATIO)]
    if len(ratio_tokens) != 2:
        raise ValueError("SEMIDETR_SAMPLE_RATIO must have two ints, e.g. 1,4")
    ratio_txt = f"[{ratio_tokens[0]}, {ratio_tokens[1]}]"
    backbone_override = ""
    if INIT_MODE == "scratch":
        backbone_override = (
            "model = dict(\n"
            "    backbone=dict(init_cfg=None, frozen_stages=-1),\n"
            "    bbox_head=dict(num_classes=1),\n"
            ")\n"
        )
    else:
        backbone_override = "model = dict(bbox_head=dict(num_classes=1))\n"

    txt = (
        f"_base_ = r'''{SEMI_DETR_BASE_CFG}'''\n\n"
        f"{backbone_override}\n"
        f"data = dict(\n"
        f"    samples_per_gpu={SAMPLES_PER_GPU},\n"
        f"    workers_per_gpu={WORKERS_PER_GPU},\n"
        f"    train=dict(\n"
        f"        sup=dict(type='CocoDataset', ann_file=r'''{ann_sup}''', img_prefix='', classes=('{class_name}',)),\n"
        f"        unsup=dict(type='CocoDataset', ann_file=r'''{ann_unsup}''', img_prefix='', classes=('{class_name}',), filter_empty_gt=False),\n"
        f"    ),\n"
        f"    val=dict(type='CocoDataset', ann_file=r'''{ann_val}''', img_prefix='', classes=('{class_name}',)),\n"
        f"    test=dict(type='CocoDataset', ann_file=r'''{ann_test}''', img_prefix='', classes=('{class_name}',)),\n"
        f"    sampler=dict(train=dict(sample_ratio={ratio_txt})),\n"
        f")\n\n"
        f"semi_wrapper = dict(train_cfg=dict(pseudo_label_initial_score_thr={PSEUDO_THRESH}, unsup_weight={UNSUP_WEIGHT}))\n"
        f"runner = dict(_delete_=True, type='IterBasedRunner', max_iters={MAX_ITERS})\n"
        f"evaluation = dict(type='SubModulesDistEvalHook', interval={EVAL_INTERVAL})\n"
        f"checkpoint_config = dict(by_epoch=False, interval={CKPT_INTERVAL}, max_keep_ckpts=5, create_symlink=False)\n"
        "log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])\n"
        f"work_dir = r'''{work_dir}'''\n"
    )
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(txt, encoding="utf-8")


def _parse_best_map50(work_dir: Path) -> dict[str, Any]:
    log_jsons = sorted(work_dir.glob("*.log.json"))
    best_m50 = None
    best_map = None
    best_iter = None
    for lp in log_jsons:
        try:
            with lp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        js = json.loads(line)
                    except Exception:
                        continue
                    candidates_m50 = [
                        js.get("student.bbox_mAP_50"),
                        js.get("bbox_mAP_50"),
                        js.get("mAP_50"),
                    ]
                    candidates_map = [
                        js.get("student.bbox_mAP"),
                        js.get("bbox_mAP"),
                        js.get("mAP"),
                    ]
                    m50 = next((float(v) for v in candidates_m50 if isinstance(v, (int, float))), None)
                    mp = next((float(v) for v in candidates_map if isinstance(v, (int, float))), None)
                    if m50 is not None and (best_m50 is None or m50 > best_m50):
                        best_m50 = m50
                        best_map = mp
                        best_iter = js.get("iter")
        except Exception:
            continue
    return {
        "best_map50": best_m50,
        "best_map": best_map,
        "best_iter": best_iter,
    }


def _run_semidetr_train(run_cfg: Path, work_dir: Path, seed: int) -> None:
    cmd = [sys.executable, "-u", str(SEMI_DETR_TRAIN_SCRIPT), str(run_cfg), "--work-dir", str(work_dir), "--launcher", "none", "--seed", str(seed)]
    env = os.environ.copy()
    py_parts = [str(SEMI_DETR_REPO), str(SEMI_DETR_REPO / "thirdparty" / "mmdetection")]
    old_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(py_parts + ([old_py] if old_py else []))
    print("[LAUNCH]", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    if DRY_RUN:
        return
    proc = subprocess.Popen(
        cmd,
        cwd=str(SEMI_DETR_REPO),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Semi-DETR training failed with exit code {rc}")


def _deps_check_cmd() -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import torch, mmcv, mmdet, detr_ssod, detr_od; "
            "print('deps_ok', torch.__version__, mmcv.__version__, mmdet.__version__)"
        ),
    ]


def _semidetr_dep_env() -> dict[str, str]:
    env = os.environ.copy()
    py_parts = [str(SEMI_DETR_REPO), str(SEMI_DETR_REPO / "thirdparty" / "mmdetection")]
    old_py = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(py_parts + ([old_py] if old_py else []))
    return env


def _install_hint() -> str:
    pyq = shlex.quote(sys.executable)
    repoq = shlex.quote(str(SEMI_DETR_REPO))
    mmcv_range = "mmcv-full>=1.3.8,<=1.4.0"
    return "\n".join(
        [
            "# Recommended: run Semi-DETR in its own env (do NOT mix with aipowmic/torch2.x).",
            "conda create -y -n semidetr python=3.8",
            "conda activate semidetr",
            "python -m pip install -U pip setuptools wheel",
            "python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html",
            "",
            "# Install once in the SAME python env used to launch this script:",
            f"{pyq} -m pip install -U pip setuptools wheel",
            f"{pyq} -m pip install -U openmim",
            f"{pyq} -m mim install \"{mmcv_range}\"",
            f"cd {repoq}/thirdparty/mmdetection && {pyq} -m pip install -e .",
            f"cd {repoq} && {pyq} -m pip install -e .",
            f"cd {repoq}/detr_od/models/utils/ops && {pyq} setup.py build install",
            "",
            "# Then launch with:",
            "export SEMIDETR_PYTHON=/work/CondaEnv/miniconda3/envs/semidetr/bin/python",
            "",
            "# Or set SEMIDETR_BOOTSTRAP_DEPS=1 to let this script attempt these steps automatically.",
        ]
    )


def _bootstrap_semidetr_python_deps() -> None:
    env = _semidetr_dep_env()
    commands: list[tuple[list[str], Path]] = [
        ([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], SEMI_DETR_REPO),
        ([sys.executable, "-m", "pip", "install", "-U", "openmim"], SEMI_DETR_REPO),
        ([sys.executable, "-m", "mim", "install", "mmcv-full>=1.3.8,<=1.4.0"], SEMI_DETR_REPO),
        ([sys.executable, "-m", "pip", "install", "-e", "."], SEMI_DETR_REPO / "thirdparty" / "mmdetection"),
        ([sys.executable, "-m", "pip", "install", "-e", "."], SEMI_DETR_REPO),
        ([sys.executable, "setup.py", "build", "install"], SEMI_DETR_REPO / "detr_od" / "models" / "utils" / "ops"),
    ]
    for cmd, cwd in commands:
        print("[BOOTSTRAP]", " ".join(shlex.quote(x) for x in cmd), f"(cwd={cwd})")
        p = subprocess.run(cmd, cwd=str(cwd), env=env)
        if p.returncode != 0:
            raise RuntimeError(
                "Dependency bootstrap failed. "
                f"Command exit code {p.returncode}: {' '.join(cmd)}\n\n"
                f"Manual install hints:\n{_install_hint()}"
            )


def _check_semidetr_python_deps() -> None:
    env = _semidetr_dep_env()
    p = subprocess.run(_deps_check_cmd(), cwd=str(SEMI_DETR_REPO), env=env, capture_output=True, text=True)
    if p.returncode == 0:
        print("[PREFLIGHT] python deps:", p.stdout.strip())
        return

    msg = (p.stdout or "") + "\n" + (p.stderr or "")
    if BOOTSTRAP_DEPS:
        print("[PREFLIGHT] Semi-DETR deps missing; SEMIDETR_BOOTSTRAP_DEPS=1 so bootstrap will be attempted.")
        _bootstrap_semidetr_python_deps()
        p2 = subprocess.run(_deps_check_cmd(), cwd=str(SEMI_DETR_REPO), env=env, capture_output=True, text=True)
        if p2.returncode == 0:
            print("[PREFLIGHT] python deps after bootstrap:", p2.stdout.strip())
            return
        msg2 = (p2.stdout or "") + "\n" + (p2.stderr or "")
        raise RuntimeError(
            "Semi-DETR dependency bootstrap ran but imports still fail.\n"
            f"Initial error:\n{msg.strip()}\n\nAfter bootstrap:\n{msg2.strip()}\n\n"
            f"Manual install hints:\n{_install_hint()}"
        )

    raise RuntimeError(
        "Semi-DETR dependencies are not importable in this environment. "
        f"Python: {sys.executable}\n"
        f"Repo: {SEMI_DETR_REPO}\n\n"
        f"Details:\n{msg.strip()}\n\n"
        f"{_install_hint()}"
    )


def _preflight() -> None:
    global SEMI_DETR_REPO, SEMI_DETR_TRAIN_SCRIPT, SEMI_DETR_BASE_CFG
    SEMI_DETR_REPO, SEMI_DETR_TRAIN_SCRIPT, SEMI_DETR_BASE_CFG = _resolve_semidetr_repo_and_cfg()
    os.environ["SEMIDETR_REPO_DIR"] = str(SEMI_DETR_REPO)
    os.environ["SEMIDETR_BASE_CONFIG"] = str(SEMI_DETR_BASE_CFG)

    _require_exists(SEMI_DETR_REPO, "SEMIDETR_REPO_DIR")
    _require_exists(SEMI_DETR_TRAIN_SCRIPT, "Semi-DETR train script")
    _require_exists(SEMI_DETR_BASE_CFG, "Semi-DETR base config")
    _require_exists(IMAGES_FALLBACK_ROOT, "IMAGES_FALLBACK_ROOT")
    ds, _ = _resolve_dataset_dir()
    _require_exists(ds, "Dataset dir")
    for split in ("train", "valid", "test"):
        p = ds / split / "_annotations.coco.json"
        _require_exists(p, f"{split} annotations")
    _check_semidetr_python_deps()
    print("[PREFLIGHT] OK")


def main() -> None:
    _preflight()
    dataset_dir, class_name = _resolve_dataset_dir()
    print(f"[SEMIDETR] python={sys.executable}")
    print(f"[SEMIDETR] runtime={RUNTIME_PROFILE} target={TARGET} class={class_name}")
    print(f"[SEMIDETR] dataset={dataset_dir}")
    print(f"[SEMIDETR] init_mode={INIT_MODE} fractions={FRACTIONS} seeds={SEEDS} fraction_seeds={FRACTION_SEEDS}")
    print(f"[SEMIDETR] repo={SEMI_DETR_REPO}")
    print(f"[SEMIDETR] base_cfg={SEMI_DETR_BASE_CFG}")
    print(f"[SEMIDETR] output_root={OUTPUT_ROOT}")
    print(f"[SEMIDETR] max_iters={MAX_ITERS} pseudo_thr={PSEUDO_THRESH} unsup_weight={UNSUP_WEIGHT}")

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = OUTPUT_ROOT / f"session_semidetr_{session_id}"
    session_root.mkdir(parents=True, exist_ok=False)

    resolved_root = build_resolved_static_dataset(
        src_dir=dataset_dir,
        dst_dir=OUTPUT_ROOT / f"{dataset_dir.name}_RESOLVED",
        images_root=IMAGES_FALLBACK_ROOT,
    )
    excluded = _collect_resolved_paths(resolved_root)
    unlabeled_pool = _build_unlabeled_pool(
        images_root=IMAGES_FALLBACK_ROOT,
        excluded=excluded,
        cache_json=session_root / "_cache" / "unlabeled_pool.json",
    )
    if not unlabeled_pool:
        raise RuntimeError("No unlabeled images found after excluding labeled/val/test images.")
    print(f"[SEMIDETR] unlabeled_pool={len(unlabeled_pool)}")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    fs_by_seed = {s: FRACTION_SEEDS[0] for s in SEEDS} if len(FRACTION_SEEDS) == 1 else {s: FRACTION_SEEDS[i] for i, s in enumerate(SEEDS)}

    run_idx = 0
    for frac in FRACTIONS:
        for seed in SEEDS:
            run_idx += 1
            fseed = int(fs_by_seed[seed])
            run_token = f"Run_{run_idx:03d}_frac_{_short_frac_tag(frac)}_fseed_{fseed}_seed_{seed}"
            run_root = session_root / run_token
            work_dir = run_root / "work_dir"
            prep_dir = run_root / "prepared_annotations"
            cfg_path = run_root / "run_config.py"
            run_root.mkdir(parents=True, exist_ok=True)
            print(f"\n[RUN] {run_token}")

            try:
                labeled_frac_dataset = build_fractional_train_split(
                    resolved_root=resolved_root,
                    frac=float(frac),
                    cache_root=OUTPUT_ROOT,
                    seed=fseed,
                )
                categories = _load_json(labeled_frac_dataset / "train" / "_annotations.coco.json").get("categories", [])
                ann_sup, ann_unsup, ann_val, ann_test, ds_report = _make_semidetr_ann_files(
                    labeled_dataset=labeled_frac_dataset,
                    run_prep_dir=prep_dir,
                    unlabeled_pool=unlabeled_pool,
                    categories=categories,
                    seed=seed,
                )
                _make_run_cfg(
                    cfg_path=cfg_path,
                    ann_sup=ann_sup,
                    ann_unsup=ann_unsup,
                    ann_val=ann_val,
                    ann_test=ann_test,
                    class_name=class_name,
                    work_dir=work_dir,
                )
                t0 = time.time()
                _run_semidetr_train(cfg_path, work_dir, seed=seed)
                seconds = round(time.time() - t0, 2)
                metric = _parse_best_map50(work_dir)
                row = {
                    "run_idx": run_idx,
                    "fraction": float(frac),
                    "fraction_seed": fseed,
                    "seed": seed,
                    "init_mode": INIT_MODE,
                    "target": class_name,
                    "model": MODEL_CLASS_NAME,
                    "max_iters": MAX_ITERS,
                    "best_map50": metric.get("best_map50"),
                    "best_map": metric.get("best_map"),
                    "best_iter": metric.get("best_iter"),
                    "labeled_images": ds_report.get("labeled_images"),
                    "unlabeled_images": ds_report.get("selected_unlabeled_images"),
                    "seconds": seconds,
                    "work_dir": str(work_dir),
                    "run_cfg": str(cfg_path),
                }
                rows.append(row)
                (run_root / "run_summary.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                print(f"[RUN][OK] mAP50={row['best_map50']} mAP={row['best_map']} sec={seconds}")
            except Exception as e:
                err = {
                    "run_idx": run_idx,
                    "fraction": frac,
                    "fraction_seed": fseed,
                    "seed": seed,
                    "error": repr(e),
                }
                errors.append(err)
                (run_root / "run_error.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
                print(f"[RUN][ERROR] {err}")
                if not CONTINUE_ON_ERROR:
                    raise

    if rows:
        rows_sorted = sorted(rows, key=lambda r: (-(r["best_map50"] or -1.0), -(r["best_map"] or -1.0)))
        with (session_root / "leaderboard.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
            w.writeheader()
            for r in rows_sorted:
                w.writerow(r)
        best = rows_sorted[0]
    else:
        best = None

    final = {
        "session_root": str(session_root),
        "runtime_profile": RUNTIME_PROFILE,
        "target": TARGET,
        "class_name": class_name,
        "dataset_dir": str(dataset_dir),
        "init_mode": INIT_MODE,
        "fractions": FRACTIONS,
        "seeds": SEEDS,
        "fraction_seeds": FRACTION_SEEDS,
        "max_iters": MAX_ITERS,
        "pseudo_score_thresh": PSEUDO_THRESH,
        "unsup_weight": UNSUP_WEIGHT,
        "sample_ratio": UNSUP_RATIO,
        "n_completed_runs": len(rows),
        "n_errors": len(errors),
        "best": best,
        "errors": errors,
    }
    (session_root / "FINAL_SEMIDETR_SUMMARY.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("\n[FINAL] Summary ->", session_root / "FINAL_SEMIDETR_SUMMARY.json")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
