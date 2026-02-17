# build_stationary_ovr_by_sample.py
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json, re, random, sys

# ======= USER CONFIG =======
ALL_COCO_JSON = Path(r"D:/PHD/PhdData/CellScanData/Annotation_Backups/Quality Assessment Backups/02-11-2026/annotations/instances_default.json")
IMAGES_DIR    = Path(r"D:/PHD/PhdData/CellScanData/Zoom10x - Quality Assessment_Cleaned")
OUT_ROOT      = Path(r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\SOLO_Supervised_RFDETR/Stat_Dataset")
TARGET_CLASS  = "Leucocyte"            # "Squamous Epithelial Cell" or "Leucocyte"
SPLIT         = (0.60, 0.20, 0.20)
SEED          = 42
VALID_EXTS    = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

# How to store file_name inside COCO:
#   "relative" → path relative to IMAGES_DIR with POSIX slashes (recommended)
#   "basename" → just the filename (only safe if names are unique)
FILE_NAME_MODE = "relative"
# ===========================

def nowstamp() -> str: return datetime.now().strftime("%Y%m%d-%H%M%S")
def load_json(p: Path): return json.loads(p.read_text(encoding="utf-8"))
def dump_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def build_index(coco):
    anns_by_image = defaultdict(list)
    for a in coco.get("annotations", []):
        anns_by_image[a["image_id"]].append(a)
    images_by_id = {im["id"]: im for im in coco.get("images", [])}
    cats_by_name = {c["name"]: c for c in coco.get("categories", [])}
    return images_by_id, anns_by_image, cats_by_name

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
    # Accept legacy absolute/Windows paths and try to match by relative or basename
    rel = file_name.replace("\\", "/")
    direct = (images_root / rel)
    if direct.exists(): return direct
    if rel in by_rel:   return by_rel[rel]
    base = Path(rel).name
    cands = by_name.get(base, [])
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        # Prefer shortest path as a deterministic tie-breaker
        cands.sort(key=lambda q: len(str(q)))
        return cands[0]
    raise FileNotFoundError(f"Could not uniquely resolve image path for: {file_name}")

SAMPLE_RE = re.compile(r"^sample\s*([0-9]+)", re.IGNORECASE)
def extract_sample_key(path: Path) -> str:
    # Find ".../Sample 18/..." (case-insensitive)
    for part in path.parts[::-1]:
        m = SAMPLE_RE.match(part)
        if m:
            return f"Sample {int(m.group(1))}"
    m = SAMPLE_RE.match(path.name)
    if m:
        return f"Sample {int(m.group(1))}"
    return path.parent.name or "UnknownSample"

def split_samples(target_samples: list[str], split=SPLIT, seed=42):
    rnd = random.Random(seed)
    x = target_samples[:]
    rnd.shuffle(x)
    n = len(x)
    n_train = int(n*split[0])
    n_val   = int(n*split[1])
    train = x[:n_train]
    val   = x[n_train:n_train+n_val]
    test  = x[n_train+n_val:]
    return {"train": train, "valid": val, "test": test}

def to_coco_filename(abs_path: Path, images_root: Path) -> str:
    if FILE_NAME_MODE == "basename":
        return abs_path.name
    # default: relative
    return abs_path.resolve().relative_to(images_root.resolve()).as_posix()

def main():
    if not ALL_COCO_JSON.exists():
        print(f"[ERR] COCO not found: {ALL_COCO_JSON}"); sys.exit(1)
    if not IMAGES_DIR.exists():
        print(f"[ERR] IMAGES_DIR not found: {IMAGES_DIR}"); sys.exit(1)

    coco = load_json(ALL_COCO_JSON)
    images_by_id, anns_by_image, cats_by_name = build_index(coco)
    if TARGET_CLASS not in cats_by_name:
        print(f"[ERR] TARGET_CLASS '{TARGET_CLASS}' not in categories."); sys.exit(1)
    target_id_src = cats_by_name[TARGET_CLASS]["id"]

    by_rel, by_name = index_image_paths(IMAGES_DIR)

    # Map images to samples and detect which samples contain the target
    # *** Only keep images that have at least one TARGET_CLASS annotation ***
    imgid_to_sample = {}
    sample_to_imgids = defaultdict(list)
    sample_has_target = defaultdict(bool)

    for iid, im in images_by_id.items():
        anns = anns_by_image.get(iid, [])
        has_target = any(a["category_id"] == target_id_src for a in anns)
        if not has_target:
            # Skip inactive images entirely
            continue

        try:
            p_abs = resolve_image_path(im["file_name"], IMAGES_DIR, by_rel, by_name)
        except FileNotFoundError:
            continue

        s_key = extract_sample_key(p_abs)
        imgid_to_sample[iid] = s_key
        sample_to_imgids[s_key].append(iid)
        sample_has_target[s_key] = True  # by construction this is true

    target_samples = sorted([s for s, has in sample_has_target.items() if has])
    if not target_samples:
        print("[ERR] No samples contain the target class."); sys.exit(1)

    split = split_samples(target_samples, split=SPLIT, seed=SEED)

    # Build image lists per split (only active images from each selected sample)
    img_ids_per_split = {k: [] for k in ("train","valid","test")}
    for sp, samples in split.items():
        for s in samples:
            img_ids_per_split[sp].extend(sample_to_imgids[s])

    # Build a timestamp so we never overwrite old datasets
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Dataset name now includes timestamp
    ds_name = f"QA-2025v1_{TARGET_CLASS.replace(' ', '')}_OVR_{timestamp}"

    out_dir = OUT_ROOT / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single OVR category id = 0
    new_categories = [{"id": 0, "name": TARGET_CLASS, "supercategory": "none"}]

    def write_split(sp):
        images, annotations = [], []
        keep_ids = set(img_ids_per_split[sp])
        for iid in keep_ids:
            im_src = images_by_id[iid]
            p_abs  = resolve_image_path(im_src["file_name"], IMAGES_DIR, by_rel, by_name)

            # We know these have at least one target box, but we re-filter for safety
            target_anns = [a for a in anns_by_image.get(iid, []) if a["category_id"] == target_id_src]
            if not target_anns:
                continue

            images.append({
                "id": im_src["id"],
                "file_name": to_coco_filename(p_abs, IMAGES_DIR),  # <— PORTABLE
                "width": im_src["width"],
                "height": im_src["height"],
            })
            for a in target_anns:
                x, y, w, h = a["bbox"]
                w = float(max(1.0, w))
                h = float(max(1.0, h))
                annotations.append({
                    "id": a["id"],
                    "image_id": iid,
                    "category_id": 0,
                    "bbox": [float(x), float(y), w, h],
                    "area": float(max(1.0, w*h)),
                    "iscrowd": int(a.get("iscrowd", 0)),
                })

        split_dir = out_dir / sp
        split_dir.mkdir(parents=True, exist_ok=True)
        dump_json(split_dir / "_annotations.coco.json", {
            "images": images,
            "annotations": annotations,
            "categories": new_categories,
        })

        # Return set of actually kept image IDs for summary
        return {im["id"] for im in images}

    kept_ids_per_split = {}
    for sp in ("train", "valid", "test"):
        kept_ids_per_split[sp] = write_split(sp)

    # Summary for reproducibility
    def count_target(keep_ids):
        c = 0
        for iid in keep_ids:
            c += sum(1 for a in anns_by_image.get(iid, []) if a["category_id"] == target_id_src)
        return c

    summary = {
        "dataset_name": ds_name,
        "source_coco": str(ALL_COCO_JSON),
        "images_root": str(IMAGES_DIR.resolve().as_posix()),  # <— where relative paths are rooted
        "target_class": TARGET_CLASS,
        "seed": SEED,
        "split_ratio": {"train": SPLIT[0], "valid": SPLIT[1], "test": SPLIT[2]},
        "file_name_mode": FILE_NAME_MODE,
        "samples": {k: sorted(v) for k, v in split.items()},
        "counts": {
            k: {
                "n_samples": len(split[k]),
                # use actually kept active images:
                "n_images": len(kept_ids_per_split[k]),
                "n_target_boxes": count_target(kept_ids_per_split[k]),
            } for k in ("train", "valid", "test")
        },
    }
    dump_json(out_dir / "split_summary.json", summary)
    print(f"[OK] Wrote stationary OVR dataset: {out_dir}")
    print(json.dumps(summary["counts"], indent=2))

if __name__ == "__main__":
    main()
