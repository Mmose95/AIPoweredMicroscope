# split_by_sample_class_balanced_constrained.py
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import json, random, sys, re, csv
from inspect import signature
import os

from rfdetr import RFDETRSmall, RFDETRLarge, RFDETRMedium

import glob, os.path as op

# -------- UCloud user base detection (AAU/SDU layouts) --------
def _detect_user_base() -> str | None:
    # AAU style: /work/Member Files: <User#id>
    aau = glob.glob("/work/Member Files:*")
    if aau:
        return op.basename(aau[0])
    # SDU style: /work/<User#id>
    sdu = [d for d in glob.glob("/work/*#*") if op.isdir(d)]
    return op.basename(sdu[0]) if sdu else None

USER_BASE_DIR = os.environ.get("USER_BASE_DIR") or _detect_user_base()
if not USER_BASE_DIR:
    raise RuntimeError("Could not determine USER_BASE_DIR on UCloud.")
os.environ["USER_BASE_DIR"] = USER_BASE_DIR  # make it visible to subprocesses
print("USER_BASE_DIR =", USER_BASE_DIR)

# Root of your workspace on UCloud
WORK_ROOT = Path("/work") / USER_BASE_DIR

# -------- Helpers for clean env overrides --------
def env_path(name: str, default: Path) -> Path:
    v = os.getenv(name, "").strip()
    return Path(v) if v else default

# Adjust the date folder once per run (or override via env)
ANNOT_DATE = os.getenv("ANNOT_DATE", "28-10-2025")

# -------- Final resolved paths (with sensible defaults) --------
ALL_COCO_JSON = env_path(
    "ALL_COCO_JSON",
    WORK_ROOT / "CellScanData" / "Annotation_Backups" / "Quality Assessment Backups" / ANNOT_DATE / "annotations" / "instances_default.json",
)

IMAGES_DIR = env_path(
    "IMAGES_DIR",
    WORK_ROOT / "CellScanData" / "Zoom10x - Quality Assessment_Cleaned",
)

OUT_ROOT = env_path(
    "OUT_ROOT",
    WORK_ROOT / "RFDETR_SOLO_OUTPUT"
)

# -------- Sanity checks / setup --------
if not ALL_COCO_JSON.exists():
    raise FileNotFoundError(f"COCO file not found: {ALL_COCO_JSON}")
if not IMAGES_DIR.exists():
    raise FileNotFoundError(f"Images dir not found: {IMAGES_DIR}")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

print("[PATHS]")
print("  COCO:", ALL_COCO_JSON)
print("  IMGS:", IMAGES_DIR)
print("  OUT :", OUT_ROOT)

# ========================

# Split targets
SPLIT = (0.70, 0.15, 0.15)     # train / val / test
SEED  = 42
KEEP_EMPTY = False
VALID_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

# Balancing weights
ALPHA_SIZE = 0.70   # image-count deviation
BETA_CLASS = 0.30   # class-count deviation
EPS = 1e-9

# Exclude classes by NAME (e.g., zero-annotation class or classes you don't want balanced)
EXCLUDE_CLASS_NAMES = {"Cylindrical Epithelial Cell"}

# One-vs-rest targets (auto-pruned to only those present in data)
TARGET_SPECS = [
    {"name": "Squamous Epithelial Cell", "suffix": "Epithelial"},
    {"name": "Leucocyte",                "suffix": "Leucocyte"},
]

# ---- DDP helpers ----
import torch
import torch.distributed as dist

# --- replace your DDP helpers with this minimal version ----
import torch, os

def ddp_active() -> bool:
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except Exception:
        return False

def world_size() -> int:
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except Exception:
        return 1

def is_main() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def ddp_setup():
    """
    IMPORTANT: Do NOT init the process group here.
    RFDETR will call torch.distributed.init_process_group() internally.
    We only set the CUDA device so DataLoader pinning etc. is correct.
    """
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def ddp_barrier():
    # defer to RFDETR's process group after it initializes; but we can safely
    # use torch.distributed.barrier() only if it's already initialized.
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def ddp_broadcast_str(val: str | None) -> str:
    """Broadcast a short string (e.g., timestamp) from rank-0 to all ranks."""
    if not ddp_active():
        return val if val is not None else ""
    obj_list = [val]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

def make_out_dir_with_stamp(root: Path, stamp: str, suffix="Base_AllClasses") -> Path:
    out = root / f"dataset_coco_splits_{stamp}_{suffix}"
    if is_main():
        out.mkdir(parents=True, exist_ok=False)
    ddp_barrier()
    return out

# ---------- small utils ----------
def nowstamp(): return datetime.now().strftime('%Y%m%d-%H%M%S')

def load_coco(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_index(coco):
    anns_by_image = defaultdict(list)
    for a in coco.get("annotations", []):
        anns_by_image[a["image_id"]].append(a)
    images_by_id = {im["id"]: im for im in coco.get("images", [])}
    categories   = coco.get("categories", [])
    return images_by_id, anns_by_image, categories

def nonempty_images(images_by_id, anns_by_image):
    return [im for im in images_by_id.values() if len(anns_by_image.get(im["id"], [])) > 0]

def index_image_paths(root: Path):
    by_rel, by_name = {}, defaultdict(list)
    r = root.resolve()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            rel = p.resolve().relative_to(r).as_posix()
            by_rel[rel] = p
            by_name[p.name].append(p)
    return by_rel, by_name

def resolve_image_path(file_name: str, images_root: Path, by_rel: dict, by_name: dict) -> Path:
    rel = file_name.replace("\\", "/")
    direct = (images_root / rel)
    if direct.exists(): return direct
    if rel in by_rel:   return by_rel[rel]
    base = Path(rel).name
    if base in by_name:
        cands = by_name[base]
        if len(cands) == 1: return cands[0]
        raise FileNotFoundError(
            f"Ambiguous basename '{base}' in multiple folders under {images_root}:\n" +
            "\n".join(f"  - {c}" for c in cands)
        )
    raise FileNotFoundError(f"Could not resolve image '{file_name}' under {images_root}")

# ---------- sample key extraction ----------
SAMPLE_RE = re.compile(r"^sample\s*([0-9]+)", re.IGNORECASE)
def extract_sample_key(path: Path) -> str:
    for part in path.parts[::-1]:
        m = SAMPLE_RE.match(part)
        if m: return f"Sample {int(m.group(1))}"
    m = SAMPLE_RE.match(path.name)
    if m: return f"Sample {int(m.group(1))}"
    return path.parent.name or "UnknownSample"

# ---------- per-sample stats ----------
def compute_per_sample_stats(kept_images, anns_by_image, images_dir, by_rel, by_name, included_cat_ids):
    """Return: sample_key -> {'img_ids': [...], 'n_img': int, 'class_counts': Counter(included only)}"""
    stats = {}
    for im in kept_images:
        src = resolve_image_path(im["file_name"], images_dir, by_rel, by_name)
        skey = extract_sample_key(src)
        if skey not in stats:
            stats[skey] = {"img_ids": [], "n_img": 0, "class_counts": Counter()}
        stats[skey]["img_ids"].append(im["id"])
        stats[skey]["n_img"] += 1
        for a in anns_by_image.get(im["id"], []):
            cid = a["category_id"]
            if cid in included_cat_ids:
                stats[skey]["class_counts"][cid] += 1
    return stats

# ---------- split with constraints ----------
def preseed_min_per_class(sample_stats, required_cls, seed=42):
    """
    Pick a minimal set of samples for VALID and TEST to ensure each required class appears >=1,
    preferring small samples that contain the missing class.
    """
    rnd = random.Random(seed)
    samples = list(sample_stats.keys())
    rnd.shuffle(samples)

    locked = {}  # sample -> split
    for split_name in ("valid","test"):
        have = Counter()
        while True:
            missing = [c for c in required_cls if have[c] == 0]
            if not missing:
                break
            cands = []
            for s in samples:
                if s in locked:
                    continue
                sc = sample_stats[s]["class_counts"]
                contributes = any(sc[c] > 0 for c in missing)
                if contributes:
                    cands.append(s)
            if not cands:
                break
            cands.sort(key=lambda s: (sample_stats[s]["n_img"], -sum(sample_stats[s]["class_counts"][c] for c in missing)))
            pick = cands[0]
            locked[pick] = split_name
            have.update(sample_stats[pick]["class_counts"])
    return locked

def split_by_sample_balanced(sample_stats, included_cat_ids, split, seed=42, locked=None):
    """
    Greedy cost minimization with optional pre-locked assignments (e.g., to ensure class presence).
    """
    rnd = random.Random(seed)
    samples = list(sample_stats.keys())
    rnd.shuffle(samples)
    locked = locked or {}

    # global totals
    total_imgs = sum(sample_stats[s]["n_img"] for s in samples)
    total_class = Counter()
    for s in samples:
        total_class.update(sample_stats[s]["class_counts"])

    # targets
    tgt_imgs = {
        "train": int(total_imgs * split[0]),
        "valid": int(total_imgs * split[1]),
        "test":  total_imgs - int(total_imgs * split[0]) - int(total_imgs * split[1]),
    }
    tgt_class = {
        "train": Counter({c: int(total_class[c] * split[0]) for c in included_cat_ids}),
        "valid": Counter({c: int(total_class[c] * split[1]) for c in included_cat_ids}),
        "test":  Counter({c: total_class[c] - int(total_class[c]*split[0]) - int(total_class[c]*split[1]) for c in included_cat_ids}),
    }

    # tallies
    assign = {}  # sample -> split
    out = {"train": [], "valid": [], "test": []}
    cur_imgs = {"train": 0, "valid": 0, "test": 0}
    cur_class = {"train": Counter(), "valid": Counter(), "test": Counter()}

    # lock pre-seeded samples first
    for s, sp in locked.items():
        assign[s] = sp
        out[sp].extend(sample_stats[s]["img_ids"])
        cur_imgs[sp] += sample_stats[s]["n_img"]
        cur_class[sp].update(sample_stats[s]["class_counts"])

    # prioritize samples rich in rare classes
    rarity = {c: 1.0 / max(1, total_class[c]) for c in included_cat_ids}
    def rarity_score(s):
        cc = sample_stats[s]["class_counts"]
        return sum(cc[c] * rarity.get(c, 0.0) for c in cc)

    # process remaining samples
    remaining = [s for s in samples if s not in assign]
    remaining.sort(key=rarity_score, reverse=True)

    def cost_if_assign(sp, s):
        n  = sample_stats[s]["n_img"]
        cc = sample_stats[s]["class_counts"]
        size_err = abs((cur_imgs[sp] + n) - tgt_imgs[sp]) / (tgt_imgs[sp] + EPS)
        cls_err = 0.0
        for c in included_cat_ids:
            after = cur_class[sp][c] + cc[c]
            cls_err += abs(after - tgt_class[sp][c]) / (tgt_class[sp][c] + 1.0)
        # discourage putting samples with zero of ALL classes into val/test unless necessary
        if sum(cc.values()) == 0 and sp != "train":
            cls_err += 10.0
        return ALPHA_SIZE * size_err + BETA_CLASS * cls_err

    for s in remaining:
        costs = [(cost_if_assign(sp, s), sp) for sp in ("train","valid","test")]
        _, best = min(costs, key=lambda t: t[0])
        assign[s] = best
        out[best].extend(sample_stats[s]["img_ids"])
        cur_imgs[best] += sample_stats[s]["n_img"]
        cur_class[best].update(sample_stats[s]["class_counts"])

    # ensure no split empty
    for sp in ("valid","test"):
        if not out[sp]:
            train_samples = [(s, sample_stats[s]["n_img"]) for s, spx in assign.items() if spx == "train"]
            if train_samples:
                smallest, _ = min(train_samples, key=lambda kv: kv[1])
                assign[smallest] = sp
                moved = set(sample_stats[smallest]["img_ids"])
                out["train"] = [iid for iid in out["train"] if iid not in moved]
                out[sp].extend(list(moved))
                # recompute tallies
                cur_imgs = {"train": 0, "valid": 0, "test": 0}
                cur_class = {"train": Counter(), "valid": Counter(), "test": Counter()}
                for ss, dst in assign.items():
                    cur_imgs[dst] += sample_stats[ss]["n_img"]
                    cur_class[dst].update(sample_stats[ss]["class_counts"])

    return out, assign, tgt_imgs, tgt_class, cur_imgs, cur_class, total_imgs, total_class

# ---------- IO ----------
def write_subset_json(name, ids, coco, images_by_id, anns_by_image,
                      out_root, images_dir, by_rel, by_name,
                      old2new: dict, new_categories: list):
    subset_dir = out_root / name
    subset_dir.mkdir(parents=True, exist_ok=False)
    ann_out = subset_dir / "_annotations.coco.json"

    images, annotations = [], []
    for iid in ids:
        # keep only anns with included classes, remap ids
        remapped_anns = []
        for a in anns_by_image.get(iid, []):
            if a["category_id"] in old2new:
                aa = dict(a)
                aa["category_id"] = old2new[a["category_id"]]
                # clamp bbox to ≥1 px
                x, y, w, h = aa["bbox"]
                w = max(1.0, float(w)); h = max(1.0, float(h))
                aa["bbox"] = [float(x), float(y), w, h]
                remapped_anns.append(aa)

        # drop images that ended up empty in the BASE (multi-class) split
        if not remapped_anns:
            continue

        im = images_by_id[iid]
        im2 = dict(im)
        src = resolve_image_path(im["file_name"], images_dir, by_rel, by_name)
        im2["file_name"] = str(src.resolve())
        images.append(im2)
        annotations.extend(remapped_anns)

    with ann_out.open("w", encoding="utf-8") as f:
        json.dump({"images": images,
                   "annotations": annotations,
                   "categories": new_categories},
                  f, ensure_ascii=False)

def write_logs(out_dir: Path, assign: dict, sample_stats: dict,
               categories: list, included_cat_ids, tgt_imgs: dict, tgt_class: dict,
               cur_imgs: dict, cur_class: dict, total_imgs: int, total_class: Counter,
               excluded_ids: set):
    cat_id2name = {c["id"]: c["name"] for c in categories}
    def named(cnt: Counter):
        return {cat_id2name[k]: int(v) for k, v in cnt.items() if k in included_cat_ids}

    summary = {
        "notes": {
            "excluded_classes": [cat_id2name[c] for c in excluded_ids],
            "class_balanced_over": [cat_id2name[c] for c in included_cat_ids],
            "constraints": "valid/test must contain >=1 instance of each included class if feasible"
        },
        "totals": {
            "images": total_imgs,
            "per_class": named(total_class)
        },
        "targets": {
            "images": tgt_imgs,
            "per_class": {sp: named(tgt_class[sp]) for sp in ("train","valid","test")}
        },
        "achieved": {
            "images": cur_imgs,
            "per_class": {sp: named(cur_class[sp]) for sp in ("train","valid","test")}
        },
        "samples_per_split": {sp: sorted([s for s, d in assign.items() if d == sp]) for sp in ("train","valid","test")}
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# ---- derive per-class filtered datasets (keep negatives) ----
def derive_one_vs_rest(base_split_dir: Path, target_name: str) -> Path:
    """
    Create a sibling folder with suffix, keeping all images but only target-class annotations (id=0).
    """
    # infer suffix
    suffix = None
    for spec in TARGET_SPECS:
        if spec["name"] == target_name:
            suffix = spec["suffix"]
            break
    if suffix is None:
        suffix = re.sub(r"\s+", "", target_name)

    derived_dir = base_split_dir.parent / (base_split_dir.name + f"_{suffix}")
    derived_dir.mkdir(parents=True, exist_ok=False)

    # read any split file to discover original target id
    probe = json.loads((base_split_dir / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    name2id = {c["name"]: c["id"] for c in probe["categories"]}
    if target_name not in name2id:
        print(f"[WARN] Target '{target_name}' not in base categories; skipping.")
        return derived_dir
    target_old_id = name2id[target_name]

    new_categories = [{"id": 0, "name": target_name, "supercategory": "none"}]

    def filter_split(part: str):
        src = base_split_dir / part / "_annotations.coco.json"
        coco = json.loads(src.read_text(encoding="utf-8"))
        images = coco["images"]
        out_anns = []
        for a in coco["annotations"]:
            if a["category_id"] == target_old_id:
                aa = dict(a); aa["category_id"] = 0
                x,y,w,h = aa["bbox"]
                aa["bbox"] = [float(x), float(y), float(max(1.0,w)), float(max(1.0,h))]
                out_anns.append(aa)
        dst_dir = derived_dir / part
        dst_dir.mkdir(parents=True, exist_ok=True)
        (dst_dir / "_annotations.coco.json").write_text(
            json.dumps({"images": images, "annotations": out_anns, "categories": new_categories}),
            encoding="utf-8"
        )

    for part in ("train","valid","test"):
        filter_split(part)

    return derived_dir

# ---------- main ----------
def main():
    if not ALL_COCO_JSON.exists():
        print(f"ERROR: COCO not found: {ALL_COCO_JSON}"); sys.exit(1)
    if not IMAGES_DIR.exists():
        print(f"ERROR: Images dir not found: {IMAGES_DIR}"); sys.exit(1)

    # ---- DDP init early ----
    ddp_setup()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ws = world_size()

    # Shared timestamp/outdir
    stamp = nowstamp() if is_main() else None
    stamp = ddp_broadcast_str(stamp)
    out_dir = make_out_dir_with_stamp(OUT_ROOT, stamp)
    if is_main():
        print(f"[OK] Output (base split): {out_dir}")

    # ===== Rank-0: split/index/write; others wait =====
    if is_main():
        coco = load_coco(ALL_COCO_JSON)
        images_by_id, anns_by_image, categories = build_index(coco)

        kept_images = nonempty_images(images_by_id, anns_by_image) if not KEEP_EMPTY else list(images_by_id.values())
        if not kept_images:
            print("ERROR: No images to split."); sys.exit(1)

        print("[INDEX] Scanning images recursively…")
        by_rel, by_name = index_image_paths(IMAGES_DIR)
        if not by_rel and not by_name:
            print(f"ERROR: No images under {IMAGES_DIR}"); sys.exit(1)

        # class sets
        cat_name2id = {c["name"]: c["id"] for c in categories}
        excluded_ids = {cat_name2id[n] for n in EXCLUDE_CLASS_NAMES if n in cat_name2id}

        # global per-class totals
        total_class = Counter()
        for im in kept_images:
            for a in anns_by_image.get(im["id"], []):
                total_class[a["category_id"]] += 1

        # include only classes with >0 annotations and not excluded
        included_cat_ids = [c["id"] for c in categories if total_class[c["id"]] > 0 and c["id"] not in excluded_ids]
        if not included_cat_ids:
            print("ERROR: No classes with annotations to balance after exclusions."); sys.exit(1)

        # Make a stable order for included classes
        included_cats = [c for c in categories if c["id"] in included_cat_ids]
        included_cats.sort(key=lambda c: c["name"])  # fixed order

        # Remap: old_id -> new contiguous id 0..K-1
        old2new = {c["id"]: i for i, c in enumerate(included_cats)}

        # include "supercategory" so rfdetr doesn't KeyError
        new_categories = [
            {"id": i, "name": c["name"], "supercategory": c.get("supercategory", "none")}
            for i, c in enumerate(included_cats)
        ]

        # per-sample stats (only included classes counted)
        sample_stats = compute_per_sample_stats(kept_images, anns_by_image, IMAGES_DIR, by_rel, by_name, included_cat_ids)

        # ---- PRESEED: ensure valid/test have >=1 of each class if feasible ----
        locked = preseed_min_per_class(sample_stats, required_cls=included_cat_ids, seed=SEED)

        # ---- Greedy assignment with balancing + penalties ----
        (splits, assign, tgt_imgs, tgt_class,
         cur_imgs, cur_class, total_imgs, _total_class_included) = split_by_sample_balanced(
            sample_stats, included_cat_ids, SPLIT, seed=SEED, locked=locked
        )

        # Write JSONs (absolute paths) — BASE (multi-class)
        for part in ("train", "valid", "test"):
            print(f"Writing {part} JSON ({len(splits[part])} images)…")
            write_subset_json(
                part, splits[part], coco, images_by_id, anns_by_image,
                out_dir, IMAGES_DIR, by_rel, by_name,
                old2new, new_categories
            )

        # Logs
        write_logs(out_dir, assign, sample_stats, categories, included_cat_ids,
                   tgt_imgs, tgt_class, cur_imgs, cur_class, total_imgs, total_class, excluded_ids)

        # Save a marker with categories for non-main ranks to read
        (out_dir / ".CATS.json").write_text(json.dumps(new_categories), encoding="utf-8")

        print(f"[DONE] Base split at: {out_dir}")
        print("  - split_summary.json")

    # Wait for split completion; non-main reads categories
    ddp_barrier()
    if not is_main():
        new_categories = json.loads((out_dir / ".CATS.json").read_text(encoding="utf-8"))

    # ===== DERIVE ONE-VS-REST DATASETS + TRAIN =====
    base_cats = [c["name"] for c in new_categories]
    active_targets = [t for t in TARGET_SPECS if t["name"] in base_cats]
    if not active_targets:
        if is_main():
            print("[WARN] No target classes found in base split; nothing to train.")
        return

    MODEL_CLS = RFDETRLarge
    RESOLUTION = 672  # 56-divisible for Large; use 640 for Medium/Small.

    for spec in active_targets:
        target_name = spec["name"]
        suffix = spec["suffix"]

        # Rank-0 derives once; everyone reads derived path from marker
        marker = out_dir / f".DERIVED_{suffix}.txt"
        if is_main():
            print(f"\n[DERIVE] Building one-vs-rest for: {target_name}")
            derived_dir = derive_one_vs_rest(out_dir, target_name)
            marker.write_text(str(derived_dir), encoding="utf-8")
        ddp_barrier()
        derived_dir = Path(marker.read_text(encoding="utf-8").strip())
        print(f"[DERIVED] {target_name} dataset at: {derived_dir}")

        # Output dir strategy
        base_run = derived_dir / "rfdetr_run"
        if is_main():
            base_run.mkdir(parents=True, exist_ok=True)
            out_train = base_run
        else:
            out_train = base_run / f"_tmp_rank{int(os.environ.get('RANK','0'))}"
            out_train.mkdir(parents=True, exist_ok=True)

        # ensure all ranks share the same Torch hub cache (so others reuse the file)
        os.environ.setdefault("TORCH_HOME", str(WORK_ROOT / ".torch_cache"))

        # ----- rank-0 warmup to avoid N-way download races -----
        from inspect import signature
        if is_main():
            # create a tiny, throwaway instance that triggers the download once
            _tmp = MODEL_CLS()
            # if download happens inside .train(), do a quick dry-run that only loads weights:
            # many impls accept something like 'epochs=0' or 'max_steps=1'; if not, skip.
            try:
                can = set(signature(_tmp.train).parameters.keys())
                kwargs = {}
                if "epochs" in can: kwargs["epochs"] = 0
                if "dataset_dir" in can: kwargs["dataset_dir"] = str(derived_dir)  # harmless
                _tmp.train(**kwargs)
            except Exception:
                pass
            del _tmp
            # leave a simple marker so non-rank-0 can proceed confidently
            (derived_dir / ".WEIGHTS_READY").write_text("ok", encoding="utf-8")

        # all ranks wait for the weights to exist in TORCH_HOME
        import time
        if not is_main():
            marker = derived_dir / ".WEIGHTS_READY"
            while not marker.exists():
                time.sleep(0.5)

        # Build kwargs safely (only pass what this rfdetr build supports)
        model = MODEL_CLS()
        sig = signature(model.train)
        can = set(sig.parameters.keys())

        per_gpu_batch = 4  # start safe; increase if VRAM allows
        TARGET_EFF_BATCH = 32

        eff_per_step = per_gpu_batch * max(ws, 1)
        grad_accum = max(1, (TARGET_EFF_BATCH + eff_per_step - 1) // eff_per_step)

        base_lr_ref = 1e-4
        lr = base_lr_ref * (per_gpu_batch * ws * grad_accum) / 32.0
        lr_encoder = 0.15 * lr  # slower backbone

        train_kwargs = dict(
            dataset_dir=str(derived_dir),
            output_dir=str(out_train),

            resolution=RESOLUTION,
            batch_size=per_gpu_batch,
            grad_accum_steps=grad_accum,
            epochs=180,

            lr=lr,
            weight_decay=5e-4,
            dropout=0.1,

            class_names=[target_name],
            num_queries=300,
            multi_scale=True,              # if your build supports it
            gradient_checkpointing=True,
            amp=True,
            num_workers=12,                # per process; tune 8–12
            early_stopping=True,
            tensorboard=True,
            pin_memory=True,
            persistent_workers=True,
            lr_schedule="cosine",
            warmup_steps=1500,
        )

        def maybe(name, value):
            if name in can:
                train_kwargs[name] = value

        # Light augmentations
        maybe("hflip_prob", 0.5); maybe("flip_prob", 0.5)
        maybe("rotation_degrees", 5)
        maybe("translate", 0.05)
        maybe("scale_range", (0.9, 1.1))
        maybe("random_affine_prob", 0.5)
        maybe("color_jitter", 0.2)
        maybe("brightness", 0.2); maybe("contrast", 0.2)
        maybe("saturation", 0.2); maybe("hue", 0.02)
        maybe("gaussian_blur_prob", 0.2)
        maybe("do_random_resize_via_padding", False)
        maybe("square_resize_div_64", True)

        # ---- DDP hints (only if RFDETR supports them)
        maybe("distributed", ddp_active())
        maybe("ddp", ddp_active())
        maybe("rank", int(os.environ.get("RANK", "0")))
        maybe("world_size", ws)
        maybe("local_rank", local_rank)
        maybe("save_only_rank0", True)

        # ---- Log hyperparameters & model name (rank-0 only)
        if is_main():
            meta_dir = base_run / "run_meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / f"train_kwargs_{suffix}.json").write_text(json.dumps(train_kwargs, indent=2), encoding="utf-8")
            model_summary = {"model_name": model.__class__.__name__, "target_class": target_name,
                             "base_split_dir": str(out_dir), "derived_dir": str(derived_dir)}
            (meta_dir / f"model_architecture_{suffix}.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")
            print(f"[LOGGED] Saved train_kwargs and model meta for {target_name}")

        if is_main():
            print(f"[TRAIN] {model.__class__.__name__} — {target_name} "
                  f"(ws={ws}, per_gpu_batch={per_gpu_batch}, grad_accum={grad_accum})")

        model.train(**train_kwargs)

        if is_main():
            print(f"[TRAIN DONE] Outputs in: {base_run}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Exiting.")
        sys.exit(1)
