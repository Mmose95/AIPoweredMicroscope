from __future__ import annotations
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from inspect import signature
import json, random, sys, re, os, glob, os.path as op

# ───────────────────────────────────────────────────────────────────────────────
# Models
from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge

# ====== YOUR PATHS / UCloud detection ======
def _detect_user_base() -> str | None:
    aau = glob.glob("/work/Member Files:*")
    if aau:
        return op.basename(aau[0])
    sdu = [d for d in glob.glob("/work/*#*") if op.isdir(d)]
    return op.basename(sdu[0]) if sdu else None

USER_BASE_DIR = os.environ.get("USER_BASE_DIR") or _detect_user_base()
if not USER_BASE_DIR:
    raise RuntimeError("Could not determine USER_BASE_DIR on UCloud.")
os.environ["USER_BASE_DIR"] = USER_BASE_DIR
print("USER_BASE_DIR =", USER_BASE_DIR)

WORK_ROOT = Path("/work") / USER_BASE_DIR

def env_path(name: str, default: Path) -> Path:
    v = os.getenv(name, "").strip()
    return Path(v) if v else default

ANNOT_DATE = os.getenv("ANNOT_DATE", "28-10-2025")

ALL_COCO_JSON = env_path(
    "ALL_COCO_JSON",
    WORK_ROOT / "CellScanData" / "Annotation_Backups" / "Quality Assessment Backups" / ANNOT_DATE / "annotations" / "instances_default.json",
)
IMAGES_DIR = env_path(
    "IMAGES_DIR",
    WORK_ROOT / "CellScanData" / "Zoom10x - Quality Assessment_Cleaned",
)
OUT_ROOT = env_path("OUT_ROOT", WORK_ROOT / "RFDETR_SOLO_OUTPUT")

# ───────────────────────────────────────────────────────────────────────────────
# Config
SPLIT = (0.70, 0.15, 0.15)
SEED = 42
KEEP_EMPTY = False
VALID_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

ALPHA_SIZE = 0.70
BETA_CLASS  = 0.30
EPS = 1e-9

EXCLUDE_CLASS_NAMES = {"Cylindrical Epithelial Cell"}

TARGET_SPECS = [
    {"name": "Squamous Epithelial Cell", "suffix": "Epithelial"},
    {"name": "Leucocyte",                "suffix": "Leucocyte"},
]

# ───────────────────────────────────────────────────────────────────────────────
# Utils (no torch.distributed anywhere)
def nowstamp() -> str: return datetime.now().strftime('%Y%m%d-%H%M%S')

def safe_out_dir(root: Path, prefix="dataset_coco_splits", suffix="Base_AllClasses") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{prefix}_{nowstamp()}_{suffix}"
    out.mkdir(parents=True, exist_ok=False)
    return out

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

SAMPLE_RE = re.compile(r"^sample\s*([0-9]+)", re.IGNORECASE)
def extract_sample_key(path: Path) -> str:
    for part in path.parts[::-1]:
        m = SAMPLE_RE.match(part)
        if m: return f"Sample {int(m.group(1))}"
    m = SAMPLE_RE.match(path.name)
    if m: return f"Sample {int(m.group(1))}"
    return path.parent.name or "UnknownSample"

def compute_per_sample_stats(kept_images, anns_by_image, images_dir, by_rel, by_name, included_cat_ids):
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

def preseed_min_per_class(sample_stats, required_cls, seed=42):
    rnd = random.Random(seed)
    samples = list(sample_stats.keys())
    rnd.shuffle(samples)
    locked = {}
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
    rnd = random.Random(seed)
    samples = list(sample_stats.keys())
    rnd.shuffle(samples)
    locked = locked or {}

    total_imgs = sum(sample_stats[s]["n_img"] for s in samples)
    total_class = Counter()
    for s in samples:
        total_class.update(sample_stats[s]["class_counts"])

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

    assign = {}
    out = {"train": [], "valid": [], "test": []}
    cur_imgs = {"train": 0, "valid": 0, "test": 0}
    cur_class = {"train": Counter(), "valid": Counter(), "test": Counter()}

    for s, sp in locked.items():
        assign[s] = sp
        out[sp].extend(sample_stats[s]["img_ids"])
        cur_imgs[sp] += sample_stats[s]["n_img"]
        cur_class[sp].update(sample_stats[s]["class_counts"])

    rarity = {c: 1.0 / max(1, total_class[c]) for c in included_cat_ids}
    def rarity_score(s):
        cc = sample_stats[s]["class_counts"]
        return sum(cc[c] * rarity.get(c, 0.0) for c in cc)

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

    for sp in ("valid","test"):
        if not out[sp]:
            train_samples = [(s, sample_stats[s]["n_img"]) for s, spx in assign.items() if spx == "train"]
            if train_samples:
                smallest, _ = min(train_samples, key=lambda kv: kv[1])
                assign[smallest] = sp
                moved = set(sample_stats[smallest]["img_ids"])
                out["train"] = [iid for iid in out["train"] if iid not in moved]
                out[sp].extend(list(moved))
    return out, assign, tgt_imgs, tgt_class, cur_imgs, cur_class, total_imgs, total_class

def write_subset_json(name, ids, coco, images_by_id, anns_by_image,
                      out_root, images_dir, by_rel, by_name,
                      old2new: dict, new_categories: list):
    subset_dir = out_root / name
    subset_dir.mkdir(parents=True, exist_ok=False)
    ann_out = subset_dir / "_annotations.coco.json"

    images, annotations = [], []
    for iid in ids:
        remapped_anns = []
        for a in anns_by_image.get(iid, []):
            if a["category_id"] in old2new:
                aa = dict(a)
                aa["category_id"] = old2new[a["category_id"]]
                x, y, w, h = aa["bbox"]
                w = max(1.0, float(w)); h = max(1.0, float(h))
                aa["bbox"] = [float(x), float(y), w, h]
                remapped_anns.append(aa)

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

def derive_one_vs_rest(base_split_dir: Path, target_name: str) -> Path:
    suffix = None
    for spec in TARGET_SPECS:
        if spec["name"] == target_name:
            suffix = spec["suffix"]; break
    if suffix is None:
        suffix = re.sub(r"\s+", "", target_name)

    derived_dir = base_split_dir.parent / (base_split_dir.name + f"_{suffix}")
    derived_dir.mkdir(parents=True, exist_ok=False)

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

# ───────────────────────────────────────────────────────────────────────────────
def main():
    if not ALL_COCO_JSON.exists():
        raise FileNotFoundError(f"COCO file not found: {ALL_COCO_JSON}")
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images dir not found: {IMAGES_DIR}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("[PATHS]")
    print("  COCO:", ALL_COCO_JSON)
    print("  IMGS:", IMAGES_DIR)
    print("  OUT :", OUT_ROOT)

    out_dir = safe_out_dir(OUT_ROOT)
    print(f"[OK] Output (base split): {out_dir}")

    coco = load_coco(ALL_COCO_JSON)
    images_by_id, anns_by_image, categories = build_index(coco)

    kept_images = nonempty_images(images_by_id, anns_by_image) if not KEEP_EMPTY else list(images_by_id.values())
    if not kept_images:
        print("ERROR: No images to split."); sys.exit(1)

    print("[INDEX] Scanning images recursively…")
    by_rel, by_name = index_image_paths(IMAGES_DIR)
    if not by_rel and not by_name:
        print(f"ERROR: No images under {IMAGES_DIR}"); sys.exit(1)

    cat_name2id = {c["name"]: c["id"] for c in categories}
    excluded_ids = {cat_name2id[n] for n in EXCLUDE_CLASS_NAMES if n in cat_name2id}

    total_class = Counter()
    for im in kept_images:
        for a in anns_by_image.get(im["id"], []):
            total_class[a["category_id"]] += 1

    included_cat_ids = [c["id"] for c in categories if total_class[c["id"]] > 0 and c["id"] not in excluded_ids]
    if not included_cat_ids:
        print("ERROR: No classes with annotations to balance after exclusions."); sys.exit(1)

    included_cats = [c for c in categories if c["id"] in included_cat_ids]
    included_cats.sort(key=lambda c: c["name"])

    old2new = {c["id"]: i for i, c in enumerate(included_cats)}
    new_categories = [
        {"id": i, "name": c["name"], "supercategory": c.get("supercategory", "none")}
        for i, c in enumerate(included_cats)
    ]

    sample_stats = compute_per_sample_stats(kept_images, anns_by_image, IMAGES_DIR, by_rel, by_name, included_cat_ids)
    locked = preseed_min_per_class(sample_stats, required_cls=included_cat_ids, seed=SEED)

    (splits, assign, tgt_imgs, tgt_class,
     cur_imgs, cur_class, total_imgs, _total_class_included) = split_by_sample_balanced(
        sample_stats, included_cat_ids, SPLIT, seed=SEED, locked=locked
    )

    for part in ("train","valid","test"):
        print(f"Writing {part} JSON ({len(splits[part])} images)…")
        write_subset_json(
            part, splits[part], coco, images_by_id, anns_by_image,
            out_dir, IMAGES_DIR, by_rel, by_name,
            old2new, new_categories
        )
    write_logs(out_dir, assign, sample_stats, categories, included_cat_ids,
               tgt_imgs, tgt_class, cur_imgs, cur_class, total_imgs, total_class, excluded_ids)

    print(f"[DONE] Base split at: {out_dir}")
    print("  - split_summary.json")

    # ── Derive one-vs-rest datasets and train one by one (single GPU)
    base_cats = [c["name"] for c in new_categories]
    active_targets = [t for t in TARGET_SPECS if t["name"] in base_cats]
    if not active_targets:
        print("[WARN] No target classes found in base split; nothing to train.")
        return

    #Deviding hyperparameter into a set for each object
    Epithelial_Hyperparameters = dict(
        MODEL_CLS=RFDETRLarge,
        RESOLUTION=672,  # epithelial scale
        EPOCHS=220,
        LR=1e-4,
        LR_ENCODER_MULT=0.15,
        BATCH=8,
        NUM_QUERIES=300,
        WARMUP_STEPS=1500,
        SCALE_RANGE=(0.9, 1.1),  # gentle zoom
        ROT_DEG=5,
        CJ=0.2,  # color jitter
        GAUSS_BLUR=0.2,
        EARLY_STOP_PATIENCE=20
    )

    Leucocyte_Hyperparameters = dict(
        MODEL_CLS=RFDETRLarge,
        RESOLUTION=672,  # make leucocytes bigger in pixels
        EPOCHS=320,  # train longer; small objs converge slower
        LR=8e-5,  # a tad lower when upscaling & training longer
        LR_ENCODER_MULT=0.10,  # keep encoder a bit more stable
        BATCH=6,  # larger images ⇒ smaller batch to avoid OOM
        NUM_QUERIES=600,  # more slots helps recall on many tiny objs
        WARMUP_STEPS=3000,
        SCALE_RANGE=(1.0, 1.6),  # bias to zoom-in (critical for tiny objs)
        ROT_DEG=7,
        CJ=0.25,
        GAUSS_BLUR=0.1,  # less blur (don’t erase tiny details)
        EARLY_STOP_PATIENCE=35
    )

    def target_profile(name: str):
        return Leucocyte_Hyperparameters if name.lower().startswith("leuco") else Epithelial_Hyperparameters

    for spec in active_targets:
        target_name = spec["name"]
        print(f"\n[DERIVE] Building one-vs-rest for: {target_name}")

        prof = target_profile(target_name)

        MODEL_CLS = prof["MODEL_CLS"]
        RESOLUTION = prof["RESOLUTION"]
        EPOCHS = prof["EPOCHS"]
        base_lr = float(os.getenv("LR", str(prof["LR"])))
        lr_encoder = prof["LR_ENCODER_MULT"] * base_lr
        per_gpu_batch = int(os.getenv("BATCH_SIZE", str(prof["BATCH"])))
        num_queries = int(os.getenv("NUM_QUERIES", str(prof["NUM_QUERIES"])))
        warmup_steps = int(os.getenv("WARMUP_STEPS", str(prof["WARMUP_STEPS"])))
        early_stop_pat = int(os.getenv("EARLY_STOP_PATIENCE", str(prof["EARLY_STOP_PATIENCE"])))

        derived_dir = derive_one_vs_rest(out_dir, target_name)
        print(f"[DERIVED] {target_name} dataset at: {derived_dir}")

        out_train = derived_dir / "rfdetr_run"
        out_train.mkdir(parents=True, exist_ok=True)

        # Build model + safe kwargs (single GPU)
        model = MODEL_CLS()
        sig = signature(model.train)
        can = set(sig.parameters.keys())

        train_kwargs = dict(
            dataset_dir=str(derived_dir),
            output_dir=str(out_train),
            resolution=RESOLUTION,
            batch_size=per_gpu_batch,
            grad_accum_steps=1,
            epochs=EPOCHS,
            lr=base_lr,
            weight_decay=5e-4,
            dropout=0.1,
            class_names=[target_name],
            num_queries=num_queries,
            multi_scale=True,
            gradient_checkpointing=True,
            amp=True,
            num_workers=int(os.getenv("NUM_WORKERS", "8")),
            early_stopping=True,
            tensorboard=True,
            pin_memory=True,
            persistent_workers=True,
            early_stop_patience=early_stop_pat
        )

        def maybe(name, value):
            if name in can:
                train_kwargs[name] = value

            # scheduler / optimizer knobs

        maybe("lr_encoder", lr_encoder)
        maybe("lr_schedule", "cosine")
        maybe("warmup_steps", warmup_steps)
        maybe("clip_grad_norm", 0.1)  # if supported: stabilizes long runs
        maybe("ema", True)  # if supported: helps small-obj recall


        # augmentation tailored per target
        scale_min, scale_max = prof["SCALE_RANGE"]
        maybe("scale_range", (scale_min, scale_max))
        maybe("rotation_degrees", prof["ROT_DEG"])
        maybe("color_jitter", prof["CJ"])
        maybe("gaussian_blur_prob", prof["GAUSS_BLUR"])
        maybe("hflip_prob", 0.5);
        maybe("flip_prob", 0.5)
        # small-obj friendly crops/padding if available in your API:
        maybe("random_resize_via_padding", True)  # keep objects from being cut off
        maybe("min_crop_size", 0.50)  # don’t crop away tiny objs
        maybe("square_resize_div_64", True)

        # early stopping patience (if available)
        maybe("early_stopping_patience", early_stop_pat)
        # and make the metric used consistent (if supported)
        maybe("early_stopping_metric", "map_50_95")
        # make early stopping check every epoch
        maybe("checkpoint_interval", 1)
        # turn on ES, use EMA for the early-stop metric, and set your patience + min_delta
        maybe("early_stopping_use_ema", True)  # your logs show max(regular, EMA); be explicit
        maybe("early_stopping_patience", 20)  # your desired patience
        maybe("early_stopping_min_delta", 0.001)  # your threshold for “improvement”
        maybe("run_test", True)  # ensure a val/test pass each epoch

        # save run meta
        meta_dir = out_train / "run_meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "train_kwargs.json").write_text(json.dumps(train_kwargs, indent=2), encoding="utf-8")

        print(f"[TRAIN] {model.__class__.__name__} — {target_name}")
        model.train(**train_kwargs)
        print(f"[TRAIN DONE] Outputs in: {out_train}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Exiting.")
        sys.exit(1)
