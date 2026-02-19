from __future__ import annotations
from pathlib import Path
from datetime import datetime
from inspect import signature
import os, json, glob, os.path as op, time, csv, re, random
from collections import defaultdict as ddict, defaultdict
from PIL import Image

# ───────────────────────────────────────────────────────────────────────────────
# TOGGLES
# ───────────────────────────────────────────────────────────────────────────────
# Which classes to run:
#  - "leu"  -> only Leucocyte
#  - "epi"  -> only Squamous Epithelial Cell
#  - "all"  -> both
PROBE_TARGET = os.environ.get("RFDETR_PROBE_TARGET", "epi").lower()

# Patchified dataset mode (same behaviour as your HPO script)
USE_PATCH_224 = bool(int(os.getenv("RFDETR_USE_PATCH_224", "1")))
PATCH_SIZE = int(os.getenv("RFDETR_PATCH_SIZE", "224"))

# Fraction of TRAIN split to use for *all* runs (must be same across ckpts)
TRAIN_FRACTION = float(os.getenv("RFDETR_TRAIN_FRACTION", "0.125"))
FRACTION_SEED  = int(os.getenv("RFDETR_FRACTION_SEED", "42"))  # must be fixed

# Static training seed for RFDETR
SEED = int(os.getenv("SEED", "42"))

print(f"[PROBE] Target classes: {PROBE_TARGET!r} (env RFDETR_PROBE_TARGET)")
print(f"[PATCH MODE] USE_PATCH_224={USE_PATCH_224}  PATCH_SIZE={PATCH_SIZE}")
print(f"[PROBE] TRAIN_FRACTION={TRAIN_FRACTION}  FRACTION_SEED={FRACTION_SEED}  SEED={SEED}")

# ───────────────────────────────────────────────────────────────────────────────
# UCloud-friendly path detection
# ───────────────────────────────────────────────────────────────────────────────
def _detect_user_base() -> str | None:
    aau = glob.glob("/work/Member Files:*")
    if aau:
        return op.basename(aau[0])
    sdu = [d for d in glob.glob("/work/*#*") if op.isdir(d)]
    return op.basename(sdu[0]) if sdu else None

USER_BASE_DIR = os.environ.get("USER_BASE_DIR") or _detect_user_base() or ""
if USER_BASE_DIR:
    os.environ["USER_BASE_DIR"] = USER_BASE_DIR
WORK_ROOT = Path("/work") / USER_BASE_DIR if USER_BASE_DIR else Path.cwd()

def env_path(name: str, default: Path) -> Path:
    v = os.getenv(name, "").strip()
    return Path(v) if v else default

# Where the **real images** live (on your drive) for resolving COCO file_name paths
IMAGES_FALLBACK_ROOT = env_path(
    "IMAGES_FALLBACK_ROOT",
    WORK_ROOT / "CellScanData" / "Zoom10x - Quality Assessment_Cleaned",
)

# ───────────────────────────────────────────────────────────────────────────────
# ====== STATIC OVR DATASETS (jsons live in repo / project tree) ======
# ───────────────────────────────────────────────────────────────────────────────
REPO_ROOT = Path.cwd()
DEFAULT_ROOT = env_path("STAT_DATASETS_ROOT", REPO_ROOT / "Stat_Dataset")

DATASET_LEUCO = Path(os.getenv("DATASET_LEUCO", str(DEFAULT_ROOT / "QA-2025v1_Leucocyte_OVR")))
DATASET_EPI   = Path(os.getenv("DATASET_EPI",   str(DEFAULT_ROOT / "QA-2025v1_SquamousEpithelialCell_OVR")))

def _autofind_dataset(root: Path, token: str) -> Path:
    cands = sorted([p for p in root.glob(f"*{token}*") if p.is_dir()])
    if not cands:
        raise FileNotFoundError(f"Could not find dataset for token '{token}' under {root}")
    return cands[-1]

if not DATASET_LEUCO.exists():
    DATASET_LEUCO = _autofind_dataset(DEFAULT_ROOT, "Leucocyte_OVR")
if not DATASET_EPI.exists():
    DATASET_EPI = _autofind_dataset(DEFAULT_ROOT, "SquamousEpithelialCell_OVR")

# ───────────────────────────────────────────────────────────────────────────────
# OUTPUT ROOT (selection runs)
# ───────────────────────────────────────────────────────────────────────────────
# Base output root (stable)
OUTPUT_BASE = env_path("OUTPUT_BASE", WORK_ROOT / "RFDETR_SOLO_OUTPUT" / "SSL_SELECTION")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# One timestamped session per script run
SESSION_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_ROOT = OUTPUT_BASE / f"session_{SESSION_ID}"
SESSION_ROOT.mkdir(parents=True, exist_ok=False)

print(f"[PROBE] Session root: {SESSION_ROOT}")

# Backwards-compatible alias used by the rest of the script
OUTPUT_ROOT = SESSION_ROOT

print(f"[PROBE] Session root: {SESSION_ROOT}")

# ───────────────────────────────────────────────────────────────────────────────
# SSL CHECKPOINT SWEEP
# ───────────────────────────────────────────────────────────────────────────────
SSL_CKPT_ROOT = env_path("SSL_CKPT_ROOT", WORK_ROOT / "SSL_Checkpoints")

# Option A: provide explicit list via env (comma-separated)
#   export SSL_CKPTS="epoch_epoch-009.ckpt,epoch_epoch-019.ckpt,last.ckpt"
_ssl_list = os.getenv("SSL_CKPTS", "").strip()

def _find_ssl_ckpts(root: Path) -> list[Path]:
    # common suffixes
    cands = []
    for ext in ("*.ckpt", "*.pth", "*.pt"):
        cands += list(root.glob(ext))

    if not cands:
        raise FileNotFoundError(f"No SSL checkpoints found in {root}")

    def sort_key(p: Path):
        name = p.name.lower()
        if "last" in name:
            return (10**9, name)
        m = re.search(r"(epoch|ep)[^\d]*(\d+)", name)
        if m:
            return (int(m.group(2)), name)
        m2 = re.search(r"(\d+)", name)
        if m2:
            return (int(m2.group(1)), name)
        return (10**8, name)

    cands.sort(key=sort_key)
    return cands

if _ssl_list:
    SSL_BACKBONES = [SSL_CKPT_ROOT / s.strip() for s in _ssl_list.split(",") if s.strip()]
else:
    SSL_BACKBONES = _find_ssl_ckpts(SSL_CKPT_ROOT)

print("[SSL BACKBONES]")
for p in SSL_BACKBONES:
    print("  ", p)

# ───────────────────────────────────────────────
# Path resolution helpers for COCO (same as your HPO script)
# ───────────────────────────────────────────────
VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
WINDOWS_PATH_RE = re.compile(r"^[a-zA-Z]:\\")  # detect old Windows-style paths

def _index_image_paths(root: Path):
    by_rel = {}
    by_name = ddict(list)
    root = root.resolve()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            rel = p.resolve().relative_to(root).as_posix()
            by_rel[rel] = p
            by_name[p.name].append(p)
    return by_rel, by_name

def _resolve_image_path(file_name: str, images_root: Path, by_rel: dict, by_name: dict) -> Path:
    rel = file_name.replace("\\", "/")
    direct = (images_root / rel)
    if direct.exists():
        return direct
    if rel in by_rel:
        return by_rel[rel]
    base = Path(rel).name
    if base in by_name:
        cands = by_name[base]
        if len(cands) == 1:
            return cands[0]
        cands.sort(key=lambda q: len(str(q)))
        return cands[0]
    if WINDOWS_PATH_RE.match(file_name):
        if base in by_name:
            cands = by_name[base]
            cands.sort(key=lambda q: len(str(q)))
            return cands[0]
    raise FileNotFoundError(f"Could not resolve image '{file_name}' under {images_root}")

def build_resolved_static_dataset(src_dir: Path, dst_dir: Path) -> Path:
    ok_marker = dst_dir / ".RESOLVED_OK"
    if ok_marker.exists():
        print(f"[RESOLVE] Using cached resolved dataset: {dst_dir}")
        return dst_dir

    if not IMAGES_FALLBACK_ROOT.exists():
        raise FileNotFoundError(
            f"IMAGES_FALLBACK_ROOT does not exist: {IMAGES_FALLBACK_ROOT}\n"
            f"Set it to your mounted CellScanData root."
        )

    print(f"[RESOLVE] Building resolved dataset for {src_dir.name} → {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    by_rel, by_name = _index_image_paths(IMAGES_FALLBACK_ROOT)

    for split in ("train", "valid", "test"):
        src_json = src_dir / split / "_annotations.coco.json"
        if not src_json.exists():
            continue

        data = json.loads(src_json.read_text(encoding="utf-8"))
        images = data.get("images", [])
        anns   = data.get("annotations", [])
        cats   = data.get("categories", [])

        out_images = []
        missing = 0

        for im in images:
            try:
                resolved = _resolve_image_path(im["file_name"], IMAGES_FALLBACK_ROOT, by_rel, by_name)
            except FileNotFoundError as e:
                missing += 1
                print(f"[RESOLVE][WARN] {e}")
                continue

            im2 = dict(im)
            im2["file_name"] = str(resolved.resolve())
            out_images.append(im2)

        valid_ids = {im["id"] for im in out_images}
        out_anns = [a for a in anns if a["image_id"] in valid_ids]

        out_split_dir = dst_dir / split
        out_split_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_split_dir / "_annotations.coco.json"
        out_json.write_text(
            json.dumps({"images": out_images, "annotations": out_anns, "categories": cats}, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[RESOLVE] {split}: kept {len(out_images)} images, {len(out_anns)} anns (missing={missing})")

    ok_marker.write_text("ok", encoding="utf-8")
    return dst_dir

# ───────────────────────────────────────────────
# FRACTIONAL TRAIN SPLIT (deterministic + cached)
# ───────────────────────────────────────────────
def build_fractional_train_split(resolved_root: Path, frac: float, cache_root: Path, seed: int = 42) -> Path:
    if frac >= 0.999:
        return resolved_root

    frac_tag = f"trainfrac_{int(frac*100):02d}_seed{seed}"
    dst_root = cache_root / f"{resolved_root.name}_{frac_tag}"
    ok_marker = dst_root / ".FRACTION_OK"

    if ok_marker.exists():
        print(f"[FRACTION] Using cached {frac_tag} dataset: {dst_root}")
        return dst_root

    print(f"[FRACTION] Building {frac_tag} dataset under {dst_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    # copy val + test JSONs as-is
    for split in ("valid", "test"):
        src = resolved_root / split / "_annotations.coco.json"
        if not src.exists():
            continue
        out_dir = dst_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "_annotations.coco.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # subsample TRAIN deterministically
    src_train = resolved_root / "train" / "_annotations.coco.json"
    if not src_train.exists():
        raise FileNotFoundError(f"No train split found at {src_train}")

    data = json.loads(src_train.read_text(encoding="utf-8"))
    images = data.get("images", [])
    anns   = data.get("annotations", [])
    cats   = data.get("categories", [])

    rng = random.Random(seed)
    n_keep = max(1, int(round(len(images) * frac)))
    indices = list(range(len(images)))
    rng.shuffle(indices)
    keep_idx = set(indices[:n_keep])

    kept_images = [im for i, im in enumerate(images) if i in keep_idx]
    kept_ids    = {im["id"] for im in kept_images}
    kept_anns   = [a for a in anns if a["image_id"] in kept_ids]

    out_train_dir = dst_root / "train"
    out_train_dir.mkdir(parents=True, exist_ok=True)
    out_train = out_train_dir / "_annotations.coco.json"
    out_train.write_text(
        json.dumps({"images": kept_images, "annotations": kept_anns, "categories": cats}, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[FRACTION] train: kept {len(kept_images)}/{len(images)} images ({frac:.3f}), anns={len(kept_anns)}")
    ok_marker.write_text("ok", encoding="utf-8")
    return dst_root

# ───────────────────────────────────────────────
# PATCHIFY DATASET (optional) — keep your existing approach
# NOTE: For brevity, patchify is not reprinted here. If you need it, paste your
# build_patchified_dataset(...) + helpers from your HPO script and reuse.
# ───────────────────────────────────────────────
# >>> If you want patch mode identical to HPO, paste:
#     - _compute_positions
#     - _patchify_split
#     - build_patchified_dataset
# and then use it below in get_or_build_probe_dataset().

def get_or_build_probe_dataset(target_name: str, dataset_dir: Path, root_out: Path) -> Path:
    """
    Build the *single* dataset that ALL checkpoints will train on for this class:
      1) resolved dataset (absolute file_name)
      2) optional patchified dataset (if you paste + enable it)
      3) deterministic fractional TRAIN split (cached)
    """
    # step 1: resolved
    resolved_cache = root_out / f"{dataset_dir.name}_RESOLVED"
    resolved_dir = build_resolved_static_dataset(dataset_dir, resolved_cache)

    # step 2: patchify (optional)
    data_dir = resolved_dir
    if USE_PATCH_224:
        # If you pasted build_patchified_dataset from your HPO script, enable this:
        # data_dir = build_patchified_dataset(resolved_root=resolved_dir, cache_root=root_out,
        #                                    patch_size=PATCH_SIZE, stride=PATCH_SIZE)
        #
        # If you *don’t* paste it, we still force resolution to 224 during training (but images remain 640).
        # That’s not what you want for “no resizing”. So: strongly recommended to paste patchify.
        print("[WARN] USE_PATCH_224=1 but patchify builder not included in this script.")
        print("       Paste your build_patchified_dataset() here for identical behaviour.")

    # step 3: fixed fractional split (this is the key for identical data)
    data_dir = build_fractional_train_split(
        resolved_root=data_dir,
        frac=TRAIN_FRACTION,
        cache_root=root_out,
        seed=FRACTION_SEED,
    )
    return data_dir

# ───────────────────────────────────────────────────────────────────────────────
# Metrics extraction (reuse your find_best_val)
# ───────────────────────────────────────────────────────────────────────────────
def find_best_val(output_dir: Path) -> dict:
    p = output_dir / "results.json"
    if p.exists():
        js = json.loads(p.read_text())
        valid = js.get("class_map", {}).get("valid", [])
        val_row = None
        for r in valid:
            if r.get("class") == "all":
                val_row = r
                break
        if val_row is None and valid:
            val_row = valid[0]
        if val_row is not None:
            return {
                "best_epoch": None,
                "map50": float(val_row.get("map@50", 0.0)),
                "map5095": float(val_row.get("map@50:95", 0.0)),
                "source": "results.json",
            }

    candidates = [
        "val_best_summary.json",
        "val_metrics.json", "metrics_val.json", "coco_eval_val.json",
        "metrics.json", "val_results.json", "results_val.json",
    ]
    for name in candidates:
        for base in (output_dir, output_dir / "eval"):
            p = base / name
            if not p.exists():
                continue
            try:
                js = json.loads(p.read_text(encoding="utf-8"))
                def pick(keys):
                    for k in keys:
                        if k in js and js[k] is not None:
                            return float(js[k])
                return {
                    "best_epoch": js.get("best") or js.get("best_epoch") or js.get("epoch"),
                    "map50":      pick(["map50","mAP50","ap50","AP50","bbox/AP50"]),
                    "map5095":    pick(["map","mAP","mAP5095","bbox/mAP"]),
                    "source":     name,
                }
            except Exception:
                continue
    return {"best_epoch": None, "map50": None, "map5095": None, "source": "not_found"}

# ───────────────────────────────────────────────────────────────────────────────
# STATIC RF-DETR TRAINING CONFIG (ONLY SSL CKPT CHANGES)
# ───────────────────────────────────────────────────────────────────────────────
STATIC_CFG = dict(
    MODEL_CLS="RFDETRLarge",
    RESOLUTION=224 if USE_PATCH_224 else 672,   # if patchify exists, set 224
    EPOCHS=40,                                 # keep fixed for fair comparison
    LR=5e-5,
    LR_ENCODER_MULT=1.0,
    BATCH=16 if USE_PATCH_224 else 4,
    WARMUP_STEPS=0,
    NUM_QUERIES=200,
    AUG_COPIES=0,
    SCALE_RANGE=(0.9, 1.1),
    ROT_DEG=5.0,
    COLOR_JITTER=0.20,
    GAUSS_BLUR=0.20,
)

NUM_WORKERS = int(os.getenv("NUM_WORKERS", "8"))

# ───────────────────────────────────────────────────────────────────────────────
def train_one_run(target_name: str, dataset_dir_effective: Path, out_dir: Path, backbone_ckpt: str) -> dict:
    """
    Single run with static params. Only backbone_ckpt changes.
    """
    from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge
    name2cls = {"RFDETRSmall": RFDETRSmall, "RFDETRMedium": RFDETRMedium, "RFDETRLarge": RFDETRLarge}
    model_cls = name2cls[STATIC_CFG["MODEL_CLS"]]

    model = model_cls()
    sig = signature(model.train)
    can = set(sig.parameters.keys())

    # Force resolution in patch mode
    resolution = STATIC_CFG["RESOLUTION"]
    if USE_PATCH_224:
        resolution = PATCH_SIZE

    kwargs = dict(
        dataset_dir=str(dataset_dir_effective),
        output_dir=str(out_dir),
        class_names=[target_name],

        resolution=resolution,
        batch_size=STATIC_CFG["BATCH"],
        grad_accum_steps=8,
        epochs=STATIC_CFG["EPOCHS"],
        lr=STATIC_CFG["LR"],
        weight_decay=5e-4,
        dropout=0.1,
        num_queries=STATIC_CFG["NUM_QUERIES"],

        multi_scale=False,
        gradient_checkpointing=True,
        amp=True,
        num_workers=min(NUM_WORKERS, os.cpu_count() or NUM_WORKERS),
        pin_memory=True,
        persistent_workers=True,

        seed=SEED,
        early_stopping=True,          # you can set False if you want strict fairness; otherwise keep True but identical
        checkpoint_interval=10,
        run_test=True,
    )

    def maybe(name, value):
        if name in can and value is not None:
            kwargs[name] = value

    # SSL backbone checkpoint
    maybe("encoder_name", backbone_ckpt)
    maybe("pretrained_backbone", backbone_ckpt)

    # resizing behaviour
    if USE_PATCH_224:
        maybe("square_resize_div_64", False)
        maybe("do_random_resize_via_padding", False)
        maybe("random_resize_via_padding", False)
    else:
        maybe("square_resize_div_64", True)
        maybe("do_random_resize_via_padding", False)
        maybe("random_resize_via_padding", False)

    meta = out_dir / "run_meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "static_cfg.json").write_text(json.dumps(STATIC_CFG, indent=2), encoding="utf-8")
    (meta / "probe_setup.json").write_text(json.dumps({
        "target_name": target_name,
        "ssl_ckpt": backbone_ckpt,
        "dataset_dir_effective": str(dataset_dir_effective),
        "train_fraction": TRAIN_FRACTION,
        "fraction_seed": FRACTION_SEED,
        "use_patch_224": USE_PATCH_224,
        "patch_size": PATCH_SIZE if USE_PATCH_224 else None,
        "seed": SEED,
    }, indent=2), encoding="utf-8")

    print(f"[TRAIN] {model.__class__.__name__} — {target_name} — SSL={Path(backbone_ckpt).name} → {out_dir}")
    model.train(**kwargs)

    best = find_best_val(out_dir)
    (out_dir / "val_best_summary.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best

# ───────────────────────────────────────────────────────────────────────────────
def run_probe_for_class(session_root: Path, target_name: str, dataset_dir: Path) -> dict:
    """
    Build the *single* dataset once, then sweep SSL checkpoints.
    """
    class_token = (
        "Epithelial" if "Epithelial" in target_name else
        "Leucocytes" if "Leucocyte" in target_name else
        target_name.replace(" ", "")
    )

    out_root = session_root / class_token  # <- includes date+timestamp via session_root
    out_root.mkdir(parents=True, exist_ok=True)

    # Build identical dataset once
    data_dir_effective = get_or_build_probe_dataset(target_name, dataset_dir, OUTPUT_ROOT)

    leaderboard = []
    for i, ckpt in enumerate(SSL_BACKBONES, start=1):
        run_dir = out_root / f"SSL_{i:02d}__{ckpt.stem}"
        run_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        best = train_one_run(
            target_name=target_name,
            dataset_dir_effective=data_dir_effective,
            out_dir=run_dir,
            backbone_ckpt=str(ckpt),
        )
        dur = round(time.time() - t0, 2)

        row = {
            "idx": i,
            "target": target_name,
            "ssl_ckpt": str(ckpt),
            "ssl_name": ckpt.name,
            "val_AP50": best.get("map50"),
            "val_mAP5095": best.get("map5095"),
            "metrics_source": best.get("source"),
            "output_dir": str(run_dir),
            "seconds": dur,
            "train_fraction": TRAIN_FRACTION,
            "fraction_seed": FRACTION_SEED,
            "seed": SEED,
            "use_patch_224": USE_PATCH_224,
            "patch_size": PATCH_SIZE if USE_PATCH_224 else None,
        }
        (run_dir / "probe_record.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        leaderboard.append(row)

        print(f"[PROBE] {target_name} — {ckpt.name} — AP50={row['val_AP50']}  mAP={row['val_mAP5095']}")

    # sort and save leaderboard
    def sort_key(r):
        a = r["val_AP50"]; b = r["val_mAP5095"]
        return (-(a if a is not None else -1), -(b if b is not None else -1))

    leaderboard.sort(key=sort_key)

    csv_path = out_root / "ssl_probe_leaderboard.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(leaderboard[0].keys()) if leaderboard else [])
        w.writeheader()
        for r in leaderboard:
            w.writerow(r)

    best_row = leaderboard[0] if leaderboard else None
    (out_root / "ssl_probe_best.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")

    return {"best": best_row, "leaderboard": leaderboard, "out_dir": str(out_root), "dataset_effective": str(data_dir_effective)}

# ───────────────────────────────────────────────────────────────────────────────
def main():
    # sanity: dataset splits exist
    for p in (DATASET_LEUCO, DATASET_EPI):
        for part in ("train", "valid"):
            if not (p / part / "_annotations.coco.json").exists():
                raise FileNotFoundError(f"Missing {part} split in {p}")

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = OUTPUT_ROOT / f"session_{session_id}"
    session_root.mkdir(parents=True, exist_ok=False)
    print(f"[PROBE] Session root: {session_root}")

    selected = []
    if PROBE_TARGET in ("leu", "all"):
        selected.append(("Leucocyte", DATASET_LEUCO))
    if PROBE_TARGET in ("epi", "all"):
        selected.append(("Squamous Epithelial Cell", DATASET_EPI))

    if not selected:
        raise RuntimeError("RFDETR_PROBE_TARGET must be one of: leu, epi, all")

    results = {}
    for target_name, ds in selected:
        results[target_name] = run_probe_for_class(session_root, target_name, ds)

    final = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "work_root": str(WORK_ROOT),
        "ssl_ckpt_root": str(SSL_CKPT_ROOT),
        "ssl_ckpts": [str(p) for p in SSL_BACKBONES],
        "static_cfg": STATIC_CFG,
        "train_fraction": TRAIN_FRACTION,
        "fraction_seed": FRACTION_SEED,
        "seed": SEED,
        "use_patch_224": USE_PATCH_224,
        "patch_size": PATCH_SIZE if USE_PATCH_224 else None,
        "results": {k: v["best"] for k, v in results.items()},
    }

    summary_path = session_root / "FINAL_SSL_PROBE_SUMMARY.json"
    summary_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print("\n[FINAL] Summary →", summary_path)
    print(json.dumps(final, indent=2))

if __name__ == "__main__":
    main()
