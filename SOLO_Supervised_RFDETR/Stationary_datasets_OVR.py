# build_stationary_ovr_by_sample.py
from __future__ import annotations
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import json, re, random, sys, os

# ======= USER CONFIG =======
ALL_COCO_JSON = Path(r"D:/PHD/PhdData/CellScanData/Annotation_Backups/Quality Assessment Backups/28-10-2025/annotations/instances_default.json")
IMAGES_DIR    = Path(r"D:/PHD/PhdData/CellScanData/Zoom10x - Quality Assessment_Cleaned")
OUT_ROOT      = Path(r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\SOLO_Supervised_RFDETR/Stat_Dataset")  # where the stationary datasets will live
TARGET_CLASS  = "Squamous Epithelial Cell"                       # ← change to "Squamous Epithelial Cell" for SEC
SPLIT         = (0.60, 0.20, 0.20)
SEED          = 42
VALID_EXTS    = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
# ===========================

def nowstamp() -> str: return datetime.now().strftime("%Y%m%d-%H%M%S")

def load_json(p: Path): return json.loads(p.read_text(encoding="utf-8"))
def dump_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

def build_index(coco):
    anns_by_image = defaultdict(list)
    for a in coco.get("annotations", []):
        anns_by_image[a["image_id"]].append(a)
    images_by_id = {im["id"]: im for im in coco.get("images", [])}
    cats_by_id   = {c["id"]: c for c in coco.get("categories", [])}
    cats_by_name = {c["name"]: c for c in coco.get("categories", [])}
    return images_by_id, anns_by_image, cats_by_id, cats_by_name

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
    if base in by_name and len(by_name[base]) == 1:
        return by_name[base][0]
    raise FileNotFoundError(f"Could not uniquely resolve image path for: {file_name}")

SAMPLE_RE = re.compile(r"^sample\s*([0-9]+)", re.IGNORECASE)
def extract_sample_key(path: Path) -> str:
    for part in path.parts[::-1]:
        m = SAMPLE_RE.match(part)
        if m: return f"Sample {int(m.group(1))}"
    m = SAMPLE_RE.match(path.name)
    if m: return f"Sample {int(m.group(1))}"
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

def main():
    if not ALL_COCO_JSON.exists():
        print(f"[ERR] COCO not found: {ALL_COCO_JSON}"); sys.exit(1)
    if not IMAGES_DIR.exists():
        print(f"[ERR] IMAGES_DIR not found: {IMAGES_DIR}"); sys.exit(1)

    coco = load_json(ALL_COCO_JSON)
    images_by_id, anns_by_image, cats_by_id, cats_by_name = build_index(coco)
    if TARGET_CLASS not in cats_by_name:
        print(f"[ERR] TARGET_CLASS '{TARGET_CLASS}' not found in categories."); sys.exit(1)
    target_id = cats_by_name[TARGET_CLASS]["id"]

    by_rel, by_name = index_image_paths(IMAGES_DIR)

    # Map image_id → sample, and detect which samples are TARGET-positive
    imgid_to_sample = {}
    sample_to_imgids = defaultdict(list)
    sample_has_target = defaultdict(bool)

    for iid, im in images_by_id.items():
        try:
            p = resolve_image_path(im["file_name"], IMAGES_DIR, by_rel, by_name)
        except FileNotFoundError:
            continue
        s_key = extract_sample_key(p)
        imgid_to_sample[iid] = s_key
        sample_to_imgids[s_key].append(iid)
        # mark if target present in this image
        if any(a["category_id"] == target_id for a in anns_by_image.get(iid, [])):
            sample_has_target[s_key] = True

    target_samples = sorted([s for s, has in sample_has_target.items() if has])
    if not target_samples:
        print("[ERR] No samples contain the target class."); sys.exit(1)

    # Split BY SAMPLE (exclusivity)
    split = split_samples(target_samples, split=SPLIT, seed=SEED)

    # Build image lists per split: include *all* images from the chosen samples
    img_ids_per_split = {k: [] for k in ("train","valid","test")}
    for sp, samples in split.items():
        for s in samples:
            img_ids_per_split[sp].extend(sample_to_imgids[s])

    # Write COCO OVR per split: images limited to selected samples; keep only target annotations
    ds_name = f"QA-2025v1_{TARGET_CLASS.replace(' ','')}_OVR"
    out_dir = OUT_ROOT / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    new_categories = [{"id": 0, "name": TARGET_CLASS, "supercategory": "none"}]

    def write_split(sp):
        images, annotations = [], []
        keep_ids = set(img_ids_per_split[sp])
        for iid in keep_ids:
            im = images_by_id[iid]
            images.append({
                "id": im["id"],
                "file_name": str(resolve_image_path(im["file_name"], IMAGES_DIR, by_rel, by_name).resolve()),
                "width": im["width"], "height": im["height"]
            })
            for a in anns_by_image.get(iid, []):
                if a["category_id"] == target_id:
                    x,y,w,h = a["bbox"]
                    annotations.append({
                        "id": a["id"],
                        "image_id": iid,
                        "category_id": 0,
                        "bbox": [float(x),float(y),float(max(1.0,w)),float(max(1.0,h))],
                        "area": float(max(1.0,w*h)),
                        "iscrowd": int(a.get("iscrowd",0))
                    })
        split_dir = out_dir / sp
        split_dir.mkdir(parents=True, exist_ok=True)
        dump_json(split_dir / "_annotations.coco.json", {
            "images": images, "annotations": annotations, "categories": new_categories
        })

    for sp in ("train","valid","test"):
        write_split(sp)

    # Summary for reproducibility
    def count_target(keep_ids):
        c=0
        for iid in keep_ids:
            c += sum(1 for a in anns_by_image.get(iid, []) if a["category_id"]==target_id)
        return c

    summary = {
        "dataset_name": ds_name,
        "source_coco": str(ALL_COCO_JSON),
        "images_root": str(IMAGES_DIR),
        "target_class": TARGET_CLASS,
        "seed": SEED,
        "split_ratio": {"train": SPLIT[0], "valid": SPLIT[1], "test": SPLIT[2]},
        "samples": {k: sorted(v) for k,v in split.items()},
        "counts": {
            k: {
                "n_samples": len(split[k]),
                "n_images": len(img_ids_per_split[k]),
                "n_target_boxes": count_target(set(img_ids_per_split[k])),
            } for k in ("train","valid","test")
        }
    }
    dump_json(out_dir / "split_summary.json", summary)
    print(f"[OK] Wrote stationary OVR dataset: {out_dir}")
    print(json.dumps(summary["counts"], indent=2))

if __name__ == "__main__":
    main()
