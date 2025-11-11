# offline_augment_materialize_dataset_tqdm.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import json, re, shutil, random, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image
import albumentations as A
import cv2
from tqdm.auto import tqdm  # ← progress bars


# ─────────────────────────── Utilities ───────────────────────────
VALID_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
WIN_ABS = re.compile(r"^[a-zA-Z]:\\")
def norm_slash(s: str) -> str: return s.replace("\\", "/")
def load_json(p: Path): return json.loads(p.read_text(encoding="utf-8"))
def dump_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def build_index(coco: dict):
    anns_by_image = defaultdict(list)
    for a in coco.get("annotations", []):
        anns_by_image[a["image_id"]].append(a)
    images_by_id = {im["id"]: im for im in coco.get("images", [])}
    categories   = coco.get("categories", [])
    return images_by_id, anns_by_image, categories

def index_image_paths(root: Path):
    by_rel, by_name = {}, defaultdict(list)
    r = root.resolve()
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            rel = p.resolve().relative_to(r).as_posix()
            by_rel[rel] = p
            by_name[p.name].append(p)
    return by_rel, by_name

def resolve_image(file_name: str, images_root: Path, by_rel: dict, by_name: dict) -> Path:
    p = Path(file_name)
    if p.exists(): return p
    alt = (images_root / file_name).resolve()
    if alt.exists(): return alt
    rel = norm_slash(file_name)
    alt2 = (images_root / rel).resolve()
    if alt2.exists(): return alt2
    base = Path(file_name).name
    cand = by_name.get(base, [])
    if len(cand) == 1: return cand[0]
    if WIN_ABS.match(file_name) and len(cand) >= 1: return cand[0]
    raise FileNotFoundError(f"Cannot resolve: {file_name}")

def pil_save_tiff_lzw(np_rgb: np.ndarray, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np_rgb, mode="RGB").save(str(dst), format="TIFF", compression="tiff_lzw")

def copy_lossless(src: Path, dst: Path, skip_existing: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and dst.exists(): return
    shutil.copyfile(src, dst)

def clip_bbox(b, w, h):
    x, y, bw, bh = b
    x = max(0.0, min(float(x),  float(w - 1)))
    y = max(0.0, min(float(y),  float(h - 1)))
    bw = max(1.0, min(float(bw), float(w - x)))
    bh = max(1.0, min(float(bh), float(h - y)))
    return [x, y, bw, bh]

def make_pipeline(preset: str):
    if preset == "leuco":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                     scale=(1.10, 1.35), rotate=(-7, 7),
                     fit_output=False, cval=0, mode=cv2.BORDER_REFLECT_101, p=0.8),
            A.GaussNoise(p=0.25),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.01, p=0.4),
            A.GaussianBlur(blur_limit=(3, 3), p=0.10),
        ], bbox_params=A.BboxParams(format="coco", label_fields=["category_id"], min_visibility=0.30))
    elif preset == "epi":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                     scale=(0.90, 1.10), rotate=(-5, 5),
                     fit_output=False, cval=0, mode=cv2.BORDER_REFLECT_101, p=0.7),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.GaussNoise(p=0.2),
        ], bbox_params=A.BboxParams(format="coco", label_fields=["category_id"], min_visibility=0.40))
    else:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
                     scale=(0.95, 1.20), rotate=(-6, 6),
                     fit_output=False, cval=0, mode=cv2.BORDER_REFLECT_101, p=0.75),
            A.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18, hue=0.02, p=0.45),
            A.GaussNoise(p=0.2),
        ], bbox_params=A.BboxParams(format="coco", label_fields=["category_id"], min_visibility=0.35))

def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(str(path)).convert("RGB"))


# ─────────────────────── Worker for TRAIN Aug ───────────────────────
@dataclass
class WorkItem:
    image_id: int
    file_name: str
    width: int
    height: int
    anns: list[dict]
    copies: int

def do_aug(item: WorkItem, images_root: str, by_rel: dict, by_name: dict,
           out_images_dir: str, preset: str, base_seed: int, skip_existing: bool):
    pipe = make_pipeline(preset)
    src = resolve_image(item.file_name, Path(images_root), by_rel, by_name)
    img = read_rgb(src)

    out = []
    orig_bboxes, orig_labels = [], []
    for a in item.anns:
        x, y, w, h = a["bbox"]
        orig_bboxes.append([float(x), float(y), float(w), float(h)])
        orig_labels.append(a["category_id"])

    for k in range(item.copies):
        seed = base_seed ^ int(item.image_id) ^ (k << 10)
        random.seed(seed); np.random.seed(seed)
        t = pipe(image=img, bboxes=orig_bboxes, category_id=orig_labels)
        aug = t["image"]; bbs = t["bboxes"]; lbs = t["category_id"]
        if not bbs:
            continue
        out_name = f"img_{item.image_id}_aug{k+1:02d}.tif"
        out_fp = Path(out_images_dir) / out_name
        if not (skip_existing and out_fp.exists()):
            pil_save_tiff_lzw(aug, out_fp)
        H, W = aug.shape[:2]
        anns = []
        for bb, lab in zip(bbs, lbs):
            anns.append({
                "category_id": int(lab),
                "bbox": clip_bbox(bb, W, H),
                "area": float(max(1.0, bb[2]*bb[3])),
                "iscrowd": 0
            })
        out.append({"file_name": f"images/{out_name}", "width": W, "height": H, "anns": anns})
    return item.image_id, out


# ───────────────────── Materializer (callable) ─────────────────────
def materialize_dataset(
    src_dataset: Path,
    images_root: Path,
    out_dataset: Path,
    copies_per_image: int = 2,
    preset: str = "generic",          # "leuco" | "epi" | "generic"
    seed: int = 42,
    workers: int = max(1, mp.cpu_count() // 2),
    skip_existing: bool = True,
) -> dict:
    """
    Build a portable dataset ready for training:
      - train/images: originals + K augmented (TIFF LZW)
      - valid/test/images: originals only
      - all COCO use relative paths (images/...)
    Returns a summary dict with basic counts.
    """
    random.seed(seed); np.random.seed(seed)

    for sp in ("train", "valid", "test"):
        coco_p = src_dataset / sp / "_annotations.coco.json"
        if not coco_p.exists():
            raise FileNotFoundError(f"Missing {coco_p}")
    if not images_root.exists():
        raise FileNotFoundError(f"images_root not found: {images_root}")

    out_dataset.mkdir(parents=True, exist_ok=True)
    by_rel, by_name = index_image_paths(images_root)

    # ------ valid/test: copy originals with progress ------
    def process_split_copy_originals(coco_path: Path, out_split_dir: Path):
        coco = load_json(coco_path)
        images_by_id, anns_by_image, categories = build_index(coco)
        out_images, out_anns = [], []
        next_ann_id = 1
        out_img_dir = out_split_dir / "images"
        out_img_dir.mkdir(parents=True, exist_ok=True)

        for im in tqdm(list(images_by_id.values()), desc=f"Copy {out_split_dir.name} originals", unit="img"):
            src = resolve_image(im["file_name"], images_root, by_rel, by_name)
            dst_name = f"orig_{im['id']}{Path(src).suffix.lower()}"
            dst = out_img_dir / dst_name
            copy_lossless(src, dst, skip_existing)
            out_images.append({
                "id": int(im["id"]),
                "file_name": f"images/{dst_name}",
                "width": int(im["width"]),
                "height": int(im["height"])
            })
            for a in anns_by_image.get(int(im["id"]), []):
                out_anns.append({
                    "id": next_ann_id,
                    "image_id": int(im["id"]),
                    "category_id": int(a["category_id"]),
                    "bbox": [float(x) for x in a["bbox"]],
                    "area": float(max(1.0, a["bbox"][2]*a["bbox"][3])),
                    "iscrowd": int(a.get("iscrowd", 0))
                })
                next_ann_id += 1
        return out_images, out_anns, categories

    for split in ("valid", "test"):
        out_split = out_dataset / split
        ims, anns, cats = process_split_copy_originals(src_dataset / split / "_annotations.coco.json", out_split)
        dump_json(out_split / "_annotations.coco.json", {"images": ims, "annotations": anns, "categories": cats})

    # ------ train: originals + augs ------
    coco_train = load_json(src_dataset / "train" / "_annotations.coco.json")
    images_by_id, anns_by_image, categories = build_index(coco_train)

    train_out = out_dataset / "train"
    train_img_dir = train_out / "images"
    train_img_dir.mkdir(parents=True, exist_ok=True)

    # Originals with progress
    train_images, train_annotations = [], []
    next_ann_id = 1
    for im in tqdm(list(images_by_id.values()), desc="Copy train originals", unit="img"):
        src_im = resolve_image(im["file_name"], images_root, by_rel, by_name)
        dst_name = f"orig_{im['id']}{Path(src_im).suffix.lower()}"
        dst = train_img_dir / dst_name
        copy_lossless(src_im, dst, skip_existing)
        train_images.append({
            "id": int(im["id"]),
            "file_name": f"images/{dst_name}",
            "width": int(im["width"]),
            "height": int(im["height"])
        })
        for a in anns_by_image.get(int(im["id"]), []):
            train_annotations.append({
                "id": next_ann_id,
                "image_id": int(im["id"]),
                "category_id": int(a["category_id"]),
                "bbox": [float(x) for x in a["bbox"]],
                "area": float(max(1.0, a["bbox"][2]*a["bbox"][3])),
                "iscrowd": int(a.get("iscrowd", 0))
            })
            next_ann_id += 1

    # Augs (parallel) with progress
    items = [
        WorkItem(
            image_id=int(im["id"]),
            file_name=str(im["file_name"]),
            width=int(im["width"]), height=int(im["height"]),
            anns=[dict(a) for a in anns_by_image.get(int(im["id"]), [])],
            copies=copies_per_image,
        )
        for im in images_by_id.values()
    ]

    aug_images, aug_anns = [], []
    next_img_id = 1_000_000
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(
                do_aug, it, str(images_root), by_rel, by_name,
                str(train_img_dir), preset, int(seed), bool(skip_existing)
            )
            for it in items
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generate augs", unit="img"):
            iid, payloads = fut.result()
            for p in payloads:
                img_id = next_img_id; next_img_id += 1
                aug_images.append({
                    "id": img_id,
                    "file_name": p["file_name"],
                    "width": int(p["width"]), "height": int(p["height"])
                })
                for a in p["anns"]:
                    aug_anns.append({
                        "id": next_ann_id,
                        "image_id": img_id,
                        "category_id": int(a["category_id"]),
                        "bbox": [float(x) for x in a["bbox"]],
                        "area": float(a["area"]),
                        "iscrowd": int(a["iscrowd"])
                    })
                    next_ann_id += 1

    # Write train COCO
    train_images.extend(aug_images)
    train_annotations.extend(aug_anns)
    dump_json(train_out / "_annotations.coco.json", {
        "images": train_images, "annotations": train_annotations, "categories": categories
    })

    summary = {
        "source_dataset": str(src_dataset.resolve()),
        "images_root": str(images_root.resolve()),
        "out_dataset": str(out_dataset.resolve()),
        "preset": preset,
        "copies_per_image": copies_per_image,
        "seed": int(seed),
        "counts": {
            "train_originals": len(images_by_id),
            "train_augmented": len(aug_images),
            "valid_images": len(load_json(src_dataset / "valid" / "_annotations.coco.json")["images"]),
            "test_images": len(load_json(src_dataset / "test" / "_annotations.coco.json")["images"]),
        }
    }
    dump_json(out_dataset / "split_summary.json", summary)
    print("[DONE] Materialized dataset at:", out_dataset)
    print(json.dumps(summary["counts"], indent=2))
    return summary


# ──────────────────── PyCharm entry point ────────────────────
if __name__ == "__main__":
    # EDIT THESE AND PRESS ▶ RUN
    SRC_DATASET = Path(r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\SOLO_Supervised_RFDETR\Stat_Dataset\QA-2025v1_Leucocyte_OVR")
    IMAGES_ROOT = Path(r"D:\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment_Cleaned")
    OUT_DATASET = Path(r"D:\PHD\PhdData\CellScanData/AUG_MATERIALIZED\QA-2025v1_Leucocyte_OVR_augK2")
    PRESET      = "leuco"          # "leuco", "epi", or "generic"

    COPIES_PER_IMAGE = 2
    SEED             = 42
    WORKERS          = max(1, mp.cpu_count() // 2)
    SKIP_EXISTING    = True

    try:
        mp.set_start_method("spawn", force=True)  # Windows-safe
    except RuntimeError:
        pass

    materialize_dataset(
        src_dataset=SRC_DATASET,
        images_root=IMAGES_ROOT,
        out_dataset=OUT_DATASET,
        copies_per_image=COPIES_PER_IMAGE,
        preset=PRESET,
        seed=SEED,
        workers=WORKERS,
        skip_existing=SKIP_EXISTING,
    )
