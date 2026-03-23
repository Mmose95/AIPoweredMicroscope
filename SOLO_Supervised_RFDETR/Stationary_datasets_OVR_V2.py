# build_stationary_ovr_by_sample_v2.py
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json, re, random, sys

# ======= USER CONFIG =======
ALL_COCO_JSON = Path(r"D:/PHD/PhdData/CellScanData/Annotation_Backups/Quality Assessment Backups/23-03-2026/annotations/instances_default.json")
IMAGES_DIR = Path(r"D:/PHD/PhdData/CellScanData/Zoom10x - Quality Assessment_Cleaned")
OUT_ROOT = Path(r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\SOLO_Supervised_RFDETR/Stat_Dataset")
TARGET_CLASS = "Squamous Epithelial Cell"            # "Squamous Epithelial Cell" or "Leucocyte"
SPLIT = (0.70, 0.15, 0.15)
SEED = 42
VALID_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

# Samples reserved for the test split.
# Accepts integers (25) or strings ("Sample 25").
FORCED_TEST_SAMPLES = [25, 14, 13, 12, 15, 17, 18, 16]
SPLIT_SEARCH_TRIALS = 250

# How to store file_name inside COCO:
#   "relative" -> path relative to IMAGES_DIR with POSIX slashes (recommended)
#   "basename" -> just the filename (only safe if names are unique)
FILE_NAME_MODE = "relative"
# ===========================

SPLIT_NAMES = ("train", "valid", "test")
SPLIT_ORDER = {name: idx for idx, name in enumerate(SPLIT_NAMES)}
SAMPLE_RE = re.compile(r"^sample\s*([0-9]+)", re.IGNORECASE)


def nowstamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


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
    direct = images_root / rel
    if direct.exists():
        return direct
    if rel in by_rel:
        return by_rel[rel]
    base = Path(rel).name
    cands = by_name.get(base, [])
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        # Prefer shortest path as a deterministic tie-breaker
        cands.sort(key=lambda q: len(str(q)))
        return cands[0]
    raise FileNotFoundError(f"Could not uniquely resolve image path for: {file_name}")


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


def unique_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def normalize_sample_key(sample) -> str:
    if isinstance(sample, int):
        return f"Sample {sample}"
    text = str(sample).strip()
    if text.isdigit():
        return f"Sample {int(text)}"
    m = SAMPLE_RE.match(text)
    if m:
        return f"Sample {int(m.group(1))}"
    return text


def normalize_sample_keys(samples):
    return unique_preserve_order([normalize_sample_key(sample) for sample in samples])


def split_samples_by_count(target_samples: list[str], split=SPLIT, seed=42):
    rnd = random.Random(seed)
    x = target_samples[:]
    rnd.shuffle(x)
    n = len(x)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    train = x[:n_train]
    val = x[n_train:n_train + n_val]
    test = x[n_train + n_val:]
    return {"train": train, "valid": val, "test": test}


def sum_target_boxes(sample_names: list[str], sample_target_box_counts: dict[str, int]) -> int:
    return sum(sample_target_box_counts[s] for s in sample_names)


def score_box_assignment(current_boxes: dict[str, int], target_boxes: dict[str, float]) -> float:
    return sum((current_boxes[name] - target_boxes[name]) ** 2 for name in SPLIT_NAMES)


def build_split_plan(
    target_samples: list[str],
    sample_target_box_counts: dict[str, int],
    split=SPLIT,
    seed=42,
    forced_test_samples=None,
    trials=SPLIT_SEARCH_TRIALS,
    known_samples=None,
):
    total_target_boxes = sum_target_boxes(target_samples, sample_target_box_counts)
    desired_target_boxes = {
        name: total_target_boxes * split[idx]
        for idx, name in enumerate(SPLIT_NAMES)
    }

    forced_test_requested = normalize_sample_keys(forced_test_samples or [])
    if not forced_test_requested:
        samples = split_samples_by_count(target_samples, split=split, seed=seed)
        actual_target_boxes = {
            name: sum_target_boxes(samples[name], sample_target_box_counts)
            for name in SPLIT_NAMES
        }
        return {
            "strategy": "random_by_sample_count",
            "samples": samples,
            "forced_test_samples_requested": [],
            "forced_test_samples_used": [],
            "total_target_boxes": total_target_boxes,
            "desired_target_boxes": desired_target_boxes,
            "assignment_target_boxes": desired_target_boxes,
            "actual_target_boxes": actual_target_boxes,
        }

    target_sample_set = set(target_samples)
    known_sample_set = set(known_samples or target_samples)
    missing_forced = sorted(set(forced_test_requested) - known_sample_set)
    if missing_forced:
        raise ValueError(
            "Forced test samples were not found in the resolved dataset: "
            + ", ".join(missing_forced)
        )

    forced_test = [sample for sample in forced_test_requested if sample in target_sample_set]
    forced_test_without_targets = [
        sample for sample in forced_test_requested
        if sample in known_sample_set and sample not in target_sample_set
    ]
    forced_test_set = set(forced_test)
    remaining_samples = [sample for sample in target_samples if sample not in forced_test_set]

    fixed_target_boxes = {
        "train": 0,
        "valid": 0,
        "test": sum_target_boxes(forced_test, sample_target_box_counts),
    }
    remaining_target_boxes = total_target_boxes - fixed_target_boxes["test"]

    residual_weights = {
        name: max(desired_target_boxes[name] - fixed_target_boxes[name], 0.0)
        for name in SPLIT_NAMES
    }
    residual_weight_sum = sum(residual_weights.values())

    if remaining_target_boxes == 0:
        additional_target_boxes = {name: 0.0 for name in SPLIT_NAMES}
    elif residual_weight_sum > 0:
        additional_target_boxes = {
            name: remaining_target_boxes * residual_weights[name] / residual_weight_sum
            for name in SPLIT_NAMES
        }
    else:
        split_weight_sum = sum(split)
        additional_target_boxes = {
            name: remaining_target_boxes * split[idx] / split_weight_sum
            for idx, name in enumerate(SPLIT_NAMES)
        }

    assignment_target_boxes = {
        name: fixed_target_boxes[name] + additional_target_boxes[name]
        for name in SPLIT_NAMES
    }

    rnd = random.Random(seed)
    best_trial_key = None
    best_samples = None
    best_actual_boxes = None

    for _ in range(max(1, trials)):
        order = remaining_samples[:]
        rnd.shuffle(order)
        order.sort(key=lambda sample: sample_target_box_counts[sample], reverse=True)

        current_samples = {"train": [], "valid": [], "test": forced_test[:]}
        current_target_boxes = dict(fixed_target_boxes)

        for sample in order:
            sample_boxes = sample_target_box_counts[sample]
            best_choice = None
            best_choice_key = None

            for split_name in SPLIT_NAMES:
                trial_boxes = dict(current_target_boxes)
                trial_boxes[split_name] += sample_boxes

                trial_score = score_box_assignment(trial_boxes, assignment_target_boxes)
                overshoot = max(0.0, trial_boxes[split_name] - assignment_target_boxes[split_name])
                deficit = assignment_target_boxes[split_name] - current_target_boxes[split_name]
                choice_key = (trial_score, overshoot, -deficit, SPLIT_ORDER[split_name])

                if best_choice_key is None or choice_key < best_choice_key:
                    best_choice_key = choice_key
                    best_choice = split_name

            current_samples[best_choice].append(sample)
            current_target_boxes[best_choice] += sample_boxes

        sample_count_penalty = sum(
            (len(current_samples[name]) - len(target_samples) * split[idx]) ** 2
            for idx, name in enumerate(SPLIT_NAMES)
        )
        trial_key = (
            score_box_assignment(current_target_boxes, assignment_target_boxes),
            sample_count_penalty,
        )

        if best_trial_key is None or trial_key < best_trial_key:
            best_trial_key = trial_key
            best_samples = current_samples
            best_actual_boxes = current_target_boxes

    return {
        "strategy": "forced_test_annotation_aware",
        "samples": best_samples,
        "forced_test_samples_requested": forced_test_requested,
        "forced_test_samples_used": forced_test,
        "forced_test_samples_without_target_annotations": forced_test_without_targets,
        "total_target_boxes": total_target_boxes,
        "desired_target_boxes": desired_target_boxes,
        "assignment_target_boxes": assignment_target_boxes,
        "actual_target_boxes": best_actual_boxes,
    }


def to_coco_filename(abs_path: Path, images_root: Path) -> str:
    if FILE_NAME_MODE == "basename":
        return abs_path.name
    return abs_path.resolve().relative_to(images_root.resolve()).as_posix()


def main():
    if not ALL_COCO_JSON.exists():
        print(f"[ERR] COCO not found: {ALL_COCO_JSON}")
        sys.exit(1)
    if not IMAGES_DIR.exists():
        print(f"[ERR] IMAGES_DIR not found: {IMAGES_DIR}")
        sys.exit(1)

    coco = load_json(ALL_COCO_JSON)
    images_by_id, anns_by_image, cats_by_name = build_index(coco)
    if TARGET_CLASS not in cats_by_name:
        print(f"[ERR] TARGET_CLASS '{TARGET_CLASS}' not in categories.")
        sys.exit(1)
    target_id_src = cats_by_name[TARGET_CLASS]["id"]

    by_rel, by_name = index_image_paths(IMAGES_DIR)

    # Map images to samples and keep only images with at least one target annotation.
    sample_to_imgids = defaultdict(list)
    sample_target_box_counts = defaultdict(int)
    known_samples = set()

    for iid, im in images_by_id.items():
        try:
            p_abs = resolve_image_path(im["file_name"], IMAGES_DIR, by_rel, by_name)
        except FileNotFoundError:
            continue

        sample_key = extract_sample_key(p_abs)
        known_samples.add(sample_key)

        target_anns = [
            a for a in anns_by_image.get(iid, [])
            if a["category_id"] == target_id_src
        ]
        if not target_anns:
            continue

        sample_to_imgids[sample_key].append(iid)
        sample_target_box_counts[sample_key] += len(target_anns)

    target_samples = sorted(sample_to_imgids)
    if not target_samples:
        print("[ERR] No samples contain the target class.")
        sys.exit(1)

    try:
        split_plan = build_split_plan(
            target_samples,
            sample_target_box_counts,
            split=SPLIT,
            seed=SEED,
            forced_test_samples=FORCED_TEST_SAMPLES,
            trials=SPLIT_SEARCH_TRIALS,
            known_samples=known_samples,
        )
    except ValueError as exc:
        print(f"[ERR] {exc}")
        sys.exit(1)

    split = split_plan["samples"]

    img_ids_per_split = {name: [] for name in SPLIT_NAMES}
    for split_name, samples in split.items():
        for sample in samples:
            img_ids_per_split[split_name].extend(sample_to_imgids[sample])

    timestamp = nowstamp()
    ds_name = f"QA-2025v1_{TARGET_CLASS.replace(' ', '')}_OVR_V2_{timestamp}"

    out_dir = OUT_ROOT / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    new_categories = [{"id": 0, "name": TARGET_CLASS, "supercategory": "none"}]

    def write_split(split_name):
        images, annotations = [], []
        keep_ids = set(img_ids_per_split[split_name])
        for iid in keep_ids:
            im_src = images_by_id[iid]
            p_abs = resolve_image_path(im_src["file_name"], IMAGES_DIR, by_rel, by_name)

            target_anns = [
                a for a in anns_by_image.get(iid, [])
                if a["category_id"] == target_id_src
            ]
            if not target_anns:
                continue

            images.append({
                "id": im_src["id"],
                "file_name": to_coco_filename(p_abs, IMAGES_DIR),
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
                    "area": float(max(1.0, w * h)),
                    "iscrowd": int(a.get("iscrowd", 0)),
                })

        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        dump_json(split_dir / "_annotations.coco.json", {
            "images": images,
            "annotations": annotations,
            "categories": new_categories,
        })
        return {im["id"] for im in images}

    kept_ids_per_split = {}
    for split_name in SPLIT_NAMES:
        kept_ids_per_split[split_name] = write_split(split_name)

    def count_target(keep_ids):
        count = 0
        for iid in keep_ids:
            count += sum(
                1 for a in anns_by_image.get(iid, [])
                if a["category_id"] == target_id_src
            )
        return count

    summary = {
        "dataset_name": ds_name,
        "source_coco": str(ALL_COCO_JSON),
        "images_root": str(IMAGES_DIR.resolve().as_posix()),
        "target_class": TARGET_CLASS,
        "seed": SEED,
        "split_ratio": {"train": SPLIT[0], "valid": SPLIT[1], "test": SPLIT[2]},
        "split_strategy": split_plan["strategy"],
        "forced_test_samples_requested": split_plan["forced_test_samples_requested"],
        "forced_test_samples_used": split_plan["forced_test_samples_used"],
        "forced_test_samples_without_target_annotations": split_plan.get(
            "forced_test_samples_without_target_annotations", []
        ),
        "target_box_planning": {
            "total_target_boxes": split_plan["total_target_boxes"],
            "ratio_targets": {
                name: round(split_plan["desired_target_boxes"][name], 2)
                for name in SPLIT_NAMES
            },
            "assignment_targets": {
                name: round(split_plan["assignment_target_boxes"][name], 2)
                for name in SPLIT_NAMES
            },
            "actual_sample_assignment": {
                name: int(split_plan["actual_target_boxes"][name])
                for name in SPLIT_NAMES
            },
        },
        "file_name_mode": FILE_NAME_MODE,
        "samples": {name: sorted(values) for name, values in split.items()},
        "counts": {
            name: {
                "n_samples": len(split[name]),
                "n_images": len(kept_ids_per_split[name]),
                "n_target_boxes": count_target(kept_ids_per_split[name]),
            }
            for name in SPLIT_NAMES
        },
    }

    dump_json(out_dir / "split_summary.json", summary)
    if (
        split_plan["forced_test_samples_used"]
        and split_plan["actual_target_boxes"]["test"] > split_plan["desired_target_boxes"]["test"]
    ):
        print(
            "[WARN] Forced test samples already exceed the requested test target-box ratio; "
            "remaining samples were rebalanced across train/valid."
        )
    print(f"[OK] Wrote stationary OVR dataset: {out_dir}")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
