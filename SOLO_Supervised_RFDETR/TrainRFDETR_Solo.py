# split_by_sample_class_balanced_constrained.py
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
import json, random, sys, re, csv
from rfdetr import RFDETRSmall, RFDETRLarge, RFDETRMedium

from inspect import signature

# ====== YOUR PATHS ======
ALL_COCO_JSON = Path(r"D:\PHD\PhdData\CellScanData\Annotation_Backups\Quality Assessment Backups\27-10-2025\annotations/instances_default.json")
IMAGES_DIR    = Path(r"D:\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment_Cleaned")
OUT_ROOT      = Path(r"RFDETR_SOLO_OUTPUT")
# ========================

# Split targets
SPLIT = (0.70, 0.15, 0.15)     # train / val / test
SEED  = 42
KEEP_EMPTY = False
VALID_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

# Balancing weights
ALPHA_SIZE = 0.30   # image-count deviation
BETA_CLASS = 0.70   # class-count deviation
EPS = 1e-9

# Exclude classes by NAME (e.g., zero-annotation class or classes you don't want balanced)
EXCLUDE_CLASS_NAMES = {"Cylindrical Epithelial Cell"}

# ---------- small utils ----------
def safe_out_dir(root: Path, prefix="dataset_coco_splits"):
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
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
    Returns dict 'locked' with samples -> fixed split ('valid' or 'test').
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
            # candidates that provide ANY missing class and are not assigned yet
            cands = []
            for s in samples:
                if s in locked:
                    continue
                sc = sample_stats[s]["class_counts"]
                contributes = any(sc[c] > 0 for c in missing)
                if contributes:
                    cands.append(s)
            if not cands:
                # not feasible to cover all required classes in this split
                break
            # choose smallest candidate (fewer images) that contributes
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
            # move smallest train sample
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
                # (optional) clamp bbox to image and ensure w/h >= 1px
                x, y, w, h = aa["bbox"]
                w = max(1.0, float(w)); h = max(1.0, float(h))
                aa["bbox"] = [float(x), float(y), w, h]
                remapped_anns.append(aa)

        # drop images that ended up empty (prevents matcher headaches)
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

    # CSV per-sample
    with (out_dir / "sample_assignments.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["sample", "split", "n_images"] + [f"class:{c['name']}" for c in categories if c["id"] in included_cat_ids]
        w.writerow(header)
        for s, dst in sorted(assign.items()):
            row = [s, dst, sample_stats[s]["n_img"]]
            for c in categories:
                if c["id"] in included_cat_ids:
                    row.append(sample_stats[s]["class_counts"][c["id"]])
            w.writerow(row)

# ---------- main ----------
def main():
    if not ALL_COCO_JSON.exists():
        print(f"ERROR: COCO not found: {ALL_COCO_JSON}"); sys.exit(1)
    if not IMAGES_DIR.exists():
        print(f"ERROR: Images dir not found: {IMAGES_DIR}"); sys.exit(1)

    out_dir = safe_out_dir(OUT_ROOT)
    print(f"[OK] Output: {out_dir}")

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
    included_cats.sort(key=lambda c: c["name"])  # or any fixed rule you prefer

    # Remap: old_id -> new contiguous id 0..K-1
    # Remap: old_id -> new contiguous id 0..K-1
    old2new = {c["id"]: i for i, c in enumerate(included_cats)}

    # IMPORTANT: include "supercategory" so rfdetr doesn't KeyError
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

    # Write JSONs (absolute paths)
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

    print(f"[DONE] Sample-exclusive, class-balanced, constraint-satisfied splits at: {out_dir}")
    print("  - split_summary.json")
    print("  - sample_assignments.csv")



    out_train = out_dir / "rfdetr_run"
    out_train.mkdir(parents=True, exist_ok=False)

    class_names = [c["name"] for c in new_categories]

    model = RFDETRLarge()

    # ---- build kwargs safely (only pass what the installed rfdetr supports) ----
    sig = signature(model.train)
    can = set(sig.parameters.keys())
    train_kwargs = dict(
        dataset_dir=str(out_dir),
        output_dir=str(out_train),

        # core
        resolution=672, #672 needed for "Large" model
        batch_size=8,
        grad_accum_steps=8,
        epochs=140,
        lr=1e-4,
        weight_decay=5e-4,
        dropout=0.1,

        class_names=class_names,

        multi_scale=False,
        num_queries=300,
        gradient_checkpointing=True,

        amp=True,
        num_workers=12,
        early_stopping=True,
    )

    # ---- light augmentations (only set if available) ----
    def maybe(name, value):
        if name in can:
            train_kwargs[name] = value

    # Save hyperparameters before training
    meta_dir = out_dir / "rfdetr_run" / "run_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    (train_kwargs_path := meta_dir / "train_kwargs.json").write_text(
        json.dumps(train_kwargs, indent=2), encoding="utf-8"
    )

    # --- save model name only ---
    model_summary = {
        "model_name": model.__class__.__name__
    }

    (meta_dir / "model_architecture.json").write_text(
        json.dumps(model_summary, indent=2), encoding="utf-8"
    )

    print(f"[LOGGED] Saved train_kwargs → {train_kwargs_path}")
    print(f"[LOGGED] Saved model name → {model.__class__.__name__}")

    # horizontal flip probability
    maybe("hflip_prob", 0.5)  # common name
    maybe("flip_prob", 0.5)  # alternative

    # slight geometric jitter (safe for boxes)
    maybe("rotation_degrees", 5)  # ±5°
    maybe("translate", 0.05)  # up to 5% shift
    maybe("scale_range", (0.9, 1.1))  # 0.9–1.1
    maybe("random_affine_prob", 0.5)

    # mild color jitter
    maybe("color_jitter", 0.2)  # single knob
    maybe("brightness", 0.2);
    maybe("contrast", 0.2)
    maybe("saturation", 0.2);
    maybe("hue", 0.02)

    # tiny gaussian blur (optional)
    maybe("gaussian_blur_prob", 0.2)

    # DETR-specific resizing toggles (if present)
    maybe("do_random_resize_via_padding", False)
    maybe("square_resize_div_64", True)

    print("[TRAIN] Starting RF-DETR …")
    model.train(**train_kwargs)
    print(f"[TRAIN DONE] Outputs in: {out_train}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Exiting.");
        sys.exit(1)
