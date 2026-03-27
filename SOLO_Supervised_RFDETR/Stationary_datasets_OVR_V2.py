# build_stationary_ovr_by_sample_v2.py
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from datetime import datetime
import json, re, random, sys

# ======= USER CONFIG =======
ALL_COCO_JSON = Path(r"D:/PHD/PhdData/CellScanData/Annotation_Backups/Quality Assessment Backups/26-03-2026/annotations/instances_default.json")
IMAGES_DIR = Path(r"D:/PHD/PhdData/CellScanData/Zoom10x - Quality Assessment_Cleaned")
OUT_ROOT = Path(r"C:\Users\SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\SOLO_Supervised_RFDETR/Stat_Dataset")
TARGET_CLASSES = ["Leucocyte", "Squamous Epithelial Cell"]       # e.g. ["Leucocyte"] or ["Leucocyte", "Squamous Epithelial Cell"] Type both classes to do two-class
SPLIT = (0.60, 0.20, 0.20)
SEED = 42
VALID_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

# Samples reserved for the test split.
# Accepts integers (25) or strings ("Sample 25").
FORCED_TEST_SAMPLES = [25, 14, 13, 12, 15, 17, 18, 16] #Currently annotated 25, 14, 13, 12, 15, 17, 18, 16
SPLIT_SEARCH_TRIALS = 250
BALANCE_PER_CLASS = True

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


def normalize_target_class_names(target_classes) -> list[str]:
    if isinstance(target_classes, str):
        raw_names = [target_classes]
    else:
        raw_names = list(target_classes)
    names = unique_preserve_order(
        [str(name).strip() for name in raw_names if str(name).strip()]
    )
    if not names:
        raise ValueError("TARGET_CLASSES must contain at least one class name.")
    return names


def dataset_token_for_target_classes(target_class_names: list[str]) -> str:
    if len(target_class_names) == 1:
        return target_class_names[0].replace(" ", "")
    return "TwoClass"


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


def sum_target_boxes_by_class(
    sample_names: list[str],
    sample_target_box_counts_by_class: dict[str, dict[str, int]],
    target_class_names: list[str],
) -> dict[str, int]:
    totals = {class_name: 0 for class_name in target_class_names}
    for sample in sample_names:
        class_counts = sample_target_box_counts_by_class.get(sample, {})
        for class_name in target_class_names:
            totals[class_name] += int(class_counts.get(class_name, 0))
    return totals


def build_assignment_targets(
    total_target_boxes: int,
    split,
    fixed_target_boxes: dict[str, int],
) -> tuple[dict[str, float], dict[str, float]]:
    desired_target_boxes = {
        name: total_target_boxes * split[idx]
        for idx, name in enumerate(SPLIT_NAMES)
    }
    remaining_target_boxes = total_target_boxes - sum(fixed_target_boxes.values())

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
    return desired_target_boxes, assignment_target_boxes


def build_desired_sample_counts(total_samples: int, split) -> dict[str, int]:
    n_train = int(total_samples * split[0])
    n_valid = int(total_samples * split[1])
    n_test = total_samples - n_train - n_valid
    return {"train": n_train, "valid": n_valid, "test": n_test}


def allocate_integer_counts(total_count: int, weights: dict[str, float]) -> dict[str, int]:
    allocations = {name: 0 for name in SPLIT_NAMES}
    if total_count <= 0:
        return allocations

    positive_weights = {
        name: max(float(weights.get(name, 0.0)), 0.0)
        for name in SPLIT_NAMES
    }
    weight_sum = sum(positive_weights.values())
    if weight_sum <= 0:
        positive_weights = {name: 1.0 for name in SPLIT_NAMES}
        weight_sum = float(len(SPLIT_NAMES))

    raw_allocations = {
        name: total_count * positive_weights[name] / weight_sum
        for name in SPLIT_NAMES
    }
    allocations = {
        name: int(raw_allocations[name])
        for name in SPLIT_NAMES
    }
    remainder = total_count - sum(allocations.values())
    if remainder > 0:
        ranked_names = sorted(
            SPLIT_NAMES,
            key=lambda name: (raw_allocations[name] - allocations[name], -SPLIT_ORDER[name]),
            reverse=True,
        )
        for idx in range(remainder):
            allocations[ranked_names[idx % len(ranked_names)]] += 1
    return allocations


def build_assignment_sample_counts(
    total_samples: int,
    split,
    fixed_sample_counts: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    desired_sample_counts = build_desired_sample_counts(total_samples, split)
    remaining_samples = total_samples - sum(fixed_sample_counts.values())
    if remaining_samples < 0:
        raise ValueError(
            "Forced test samples exceed the number of available target samples."
        )

    residual_capacities = {
        name: max(desired_sample_counts[name] - fixed_sample_counts.get(name, 0), 0)
        for name in SPLIT_NAMES
    }
    if remaining_samples == 0:
        additional_counts = {name: 0 for name in SPLIT_NAMES}
    elif sum(residual_capacities.values()) > 0:
        additional_counts = allocate_integer_counts(remaining_samples, residual_capacities)
    else:
        additional_counts = allocate_integer_counts(
            remaining_samples,
            {name: split[idx] for idx, name in enumerate(SPLIT_NAMES)},
        )

    assignment_target_sample_counts = {
        name: int(fixed_sample_counts.get(name, 0) + additional_counts[name])
        for name in SPLIT_NAMES
    }
    return desired_sample_counts, assignment_target_sample_counts


def score_box_assignment(current_boxes: dict[str, int], target_boxes: dict[str, float]) -> float:
    return sum((current_boxes[name] - target_boxes[name]) ** 2 for name in SPLIT_NAMES)


def score_relative_box_assignment(
    current_boxes: dict[str, int],
    target_boxes: dict[str, float],
) -> float:
    score = 0.0
    for name in SPLIT_NAMES:
        denom = max(float(target_boxes[name]), 1.0)
        score += ((current_boxes[name] - target_boxes[name]) / denom) ** 2
    return score


def score_relative_sample_assignment(
    current_samples: dict[str, list[str]],
    target_sample_counts: dict[str, float],
) -> float:
    score = 0.0
    for name in SPLIT_NAMES:
        denom = max(float(target_sample_counts[name]), 1.0)
        score += ((len(current_samples[name]) - target_sample_counts[name]) / denom) ** 2
    return score


def build_split_plan(
    target_samples: list[str],
    sample_target_box_counts: dict[str, int],
    sample_target_box_counts_by_class=None,
    target_class_names=None,
    split=SPLIT,
    seed=42,
    forced_test_samples=None,
    trials=SPLIT_SEARCH_TRIALS,
    known_samples=None,
    balance_per_class=BALANCE_PER_CLASS,
):
    class_names = list(target_class_names or [])
    use_class_aware_balance = bool(
        balance_per_class
        and len(class_names) > 1
        and sample_target_box_counts_by_class is not None
    )

    total_target_boxes = sum_target_boxes(target_samples, sample_target_box_counts)

    forced_test_requested = normalize_sample_keys(forced_test_samples or [])
    if not forced_test_requested and not use_class_aware_balance:
        desired_target_boxes, assignment_target_boxes = build_assignment_targets(
            total_target_boxes=total_target_boxes,
            split=split,
            fixed_target_boxes={name: 0 for name in SPLIT_NAMES},
        )
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
            "balance_per_class": False,
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
    fixed_sample_counts = {
        "train": 0,
        "valid": 0,
        "test": len(forced_test),
    }
    desired_target_boxes, assignment_target_boxes = build_assignment_targets(
        total_target_boxes=total_target_boxes,
        split=split,
        fixed_target_boxes=fixed_target_boxes,
    )
    desired_sample_counts, assignment_target_sample_counts = build_assignment_sample_counts(
        total_samples=len(target_samples),
        split=split,
        fixed_sample_counts=fixed_sample_counts,
    )

    total_target_boxes_by_class = {}
    desired_target_boxes_by_class = {}
    assignment_target_boxes_by_class = {}
    fixed_target_boxes_by_class = {}
    classes_exceeding_test_target_due_to_forced_samples = []
    if use_class_aware_balance:
        total_target_boxes_by_class = sum_target_boxes_by_class(
            target_samples,
            sample_target_box_counts_by_class,
            class_names,
        )
        for class_name in class_names:
            fixed_class_boxes = {
                "train": 0,
                "valid": 0,
                "test": sum(
                    int(sample_target_box_counts_by_class.get(sample, {}).get(class_name, 0))
                    for sample in forced_test
                ),
            }
            desired_class_boxes, assignment_class_boxes = build_assignment_targets(
                total_target_boxes=total_target_boxes_by_class[class_name],
                split=split,
                fixed_target_boxes=fixed_class_boxes,
            )
            fixed_target_boxes_by_class[class_name] = fixed_class_boxes
            desired_target_boxes_by_class[class_name] = desired_class_boxes
            assignment_target_boxes_by_class[class_name] = assignment_class_boxes
            if fixed_class_boxes["test"] > desired_class_boxes["test"]:
                classes_exceeding_test_target_due_to_forced_samples.append(class_name)

    rnd = random.Random(seed)
    best_trial_key = None
    best_samples = None
    best_actual_boxes = None
    best_actual_boxes_by_class = None
    remaining_sample_capacities = {
        name: max(assignment_target_sample_counts[name] - fixed_sample_counts[name], 0)
        for name in SPLIT_NAMES
    }

    for _ in range(max(1, trials)):
        order = remaining_samples[:]
        rnd.shuffle(order)
        current_samples = {"train": [], "valid": [], "test": forced_test[:]}
        cursor = 0
        for split_name in SPLIT_NAMES:
            take_count = remaining_sample_capacities[split_name]
            current_samples[split_name].extend(order[cursor:cursor + take_count])
            cursor += take_count

        current_target_boxes = {
            name: sum_target_boxes(current_samples[name], sample_target_box_counts)
            for name in SPLIT_NAMES
        }
        if use_class_aware_balance:
            current_target_boxes_by_class = {
                class_name: {
                    split_name: sum(
                        int(sample_target_box_counts_by_class.get(sample, {}).get(class_name, 0))
                        for sample in current_samples[split_name]
                    )
                    for split_name in SPLIT_NAMES
                }
                for class_name in class_names
            }

        sample_count_penalty = sum(
            (len(current_samples[name]) - len(target_samples) * split[idx]) ** 2
            for idx, name in enumerate(SPLIT_NAMES)
        )
        sample_count_score = score_relative_sample_assignment(
            current_samples,
            assignment_target_sample_counts,
        )
        if use_class_aware_balance:
            class_score = sum(
                score_relative_box_assignment(
                    current_target_boxes_by_class[class_name],
                    assignment_target_boxes_by_class[class_name],
                )
                for class_name in class_names
            )
            trial_key = (
                sample_count_score,
                class_score,
                score_relative_box_assignment(current_target_boxes, assignment_target_boxes),
                sample_count_penalty,
            )
        else:
            trial_key = (
                sample_count_score,
                score_box_assignment(current_target_boxes, assignment_target_boxes),
                sample_count_penalty,
            )

        if best_trial_key is None or trial_key < best_trial_key:
            best_trial_key = trial_key
            best_samples = current_samples
            best_actual_boxes = current_target_boxes
            if use_class_aware_balance:
                best_actual_boxes_by_class = {
                    class_name: dict(current_target_boxes_by_class[class_name])
                    for class_name in class_names
                }

    plan = {
        "strategy": "forced_test_annotation_aware",
        "samples": best_samples,
        "forced_test_samples_requested": forced_test_requested,
        "forced_test_samples_used": forced_test,
        "forced_test_samples_without_target_annotations": forced_test_without_targets,
        "total_target_boxes": total_target_boxes,
        "desired_target_boxes": desired_target_boxes,
        "assignment_target_boxes": assignment_target_boxes,
        "actual_target_boxes": best_actual_boxes,
        "balance_per_class": use_class_aware_balance,
        "desired_sample_counts": desired_sample_counts,
        "assignment_target_sample_counts": assignment_target_sample_counts,
    }
    if use_class_aware_balance:
        plan["total_target_boxes_by_class"] = total_target_boxes_by_class
        plan["desired_target_boxes_by_class"] = desired_target_boxes_by_class
        plan["assignment_target_boxes_by_class"] = assignment_target_boxes_by_class
        plan["actual_target_boxes_by_class"] = best_actual_boxes_by_class
        plan["classes_exceeding_test_target_due_to_forced_samples"] = (
            classes_exceeding_test_target_due_to_forced_samples
        )
    return plan


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
    try:
        target_class_names = normalize_target_class_names(TARGET_CLASSES)
    except ValueError as exc:
        print(f"[ERR] {exc}")
        sys.exit(1)

    missing_target_classes = [
        class_name for class_name in target_class_names
        if class_name not in cats_by_name
    ]
    if missing_target_classes:
        print(
            "[ERR] Target classes not found in categories: "
            + ", ".join(missing_target_classes)
        )
        sys.exit(1)

    target_source_categories = [cats_by_name[name] for name in target_class_names]
    target_source_id_to_name = {
        cat["id"]: cat["name"] for cat in target_source_categories
    }
    target_source_ids = set(target_source_id_to_name)
    target_source_id_to_new_id = {
        cat["id"]: idx for idx, cat in enumerate(target_source_categories)
    }

    by_rel, by_name = index_image_paths(IMAGES_DIR)

    # Map images to samples and keep only images with at least one selected-class annotation.
    sample_to_imgids = defaultdict(list)
    sample_target_box_counts = defaultdict(int)
    sample_target_box_counts_by_class = defaultdict(lambda: defaultdict(int))
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
            if a["category_id"] in target_source_ids
        ]
        if not target_anns:
            continue

        sample_to_imgids[sample_key].append(iid)
        sample_target_box_counts[sample_key] += len(target_anns)
        for ann in target_anns:
            class_name = target_source_id_to_name[ann["category_id"]]
            sample_target_box_counts_by_class[sample_key][class_name] += 1

    target_samples = sorted(sample_to_imgids)
    if not target_samples:
        print(
            "[ERR] No samples contain the selected target classes: "
            + ", ".join(target_class_names)
        )
        sys.exit(1)

    try:
        split_plan = build_split_plan(
            target_samples,
            sample_target_box_counts,
            sample_target_box_counts_by_class=sample_target_box_counts_by_class,
            target_class_names=target_class_names,
            split=SPLIT,
            seed=SEED,
            forced_test_samples=FORCED_TEST_SAMPLES,
            trials=SPLIT_SEARCH_TRIALS,
            known_samples=known_samples,
            balance_per_class=BALANCE_PER_CLASS,
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
    ds_name = (
        f"QA-2025v1_{dataset_token_for_target_classes(target_class_names)}_OVR_V2_{timestamp}"
    )

    out_dir = OUT_ROOT / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    new_categories = [
        {
            "id": target_source_id_to_new_id[cat["id"]],
            "name": cat["name"],
            "supercategory": cat.get("supercategory", "none") or "none",
        }
        for cat in target_source_categories
    ]

    def write_split(split_name):
        images, annotations = [], []
        keep_ids = sorted(set(img_ids_per_split[split_name]))
        for iid in keep_ids:
            im_src = images_by_id[iid]
            p_abs = resolve_image_path(im_src["file_name"], IMAGES_DIR, by_rel, by_name)

            target_anns = [
                a for a in anns_by_image.get(iid, [])
                if a["category_id"] in target_source_ids
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
                    "category_id": target_source_id_to_new_id[a["category_id"]],
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

    def count_target_by_class(keep_ids):
        counts = {class_name: 0 for class_name in target_class_names}
        total = 0
        for iid in keep_ids:
            for ann in anns_by_image.get(iid, []):
                class_name = target_source_id_to_name.get(ann["category_id"])
                if class_name is None:
                    continue
                counts[class_name] += 1
                total += 1
        return total, counts

    summary = {
        "dataset_name": ds_name,
        "source_coco": str(ALL_COCO_JSON),
        "images_root": str(IMAGES_DIR.resolve().as_posix()),
        "target_class_mode": ("single_class" if len(target_class_names) == 1 else "multi_class"),
        "target_classes": target_class_names,
        "balance_per_class": bool(split_plan.get("balance_per_class", False)),
        "seed": SEED,
        "split_ratio": {"train": SPLIT[0], "valid": SPLIT[1], "test": SPLIT[2]},
        "split_strategy": split_plan["strategy"],
        "forced_test_samples_requested": split_plan["forced_test_samples_requested"],
        "forced_test_samples_used": split_plan["forced_test_samples_used"],
        "forced_test_samples_without_target_annotations": split_plan.get(
            "forced_test_samples_without_target_annotations", []
        ),
        "target_box_planning": {
            "balance_per_class": bool(split_plan.get("balance_per_class", False)),
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
        "target_sample_planning": {
            "ratio_targets": {
                name: int(split_plan["desired_sample_counts"][name])
                for name in SPLIT_NAMES
            },
            "assignment_targets": {
                name: int(split_plan["assignment_target_sample_counts"][name])
                for name in SPLIT_NAMES
            },
            "actual_sample_assignment": {
                name: len(split[name])
                for name in SPLIT_NAMES
            },
        },
        "file_name_mode": FILE_NAME_MODE,
        "samples": {name: sorted(values) for name, values in split.items()},
        "counts": {},
    }
    if len(target_class_names) == 1:
        summary["target_class"] = target_class_names[0]
    if split_plan.get("balance_per_class"):
        summary["target_box_planning"]["per_class"] = {
            class_name: {
                "total_target_boxes": int(split_plan["total_target_boxes_by_class"][class_name]),
                "ratio_targets": {
                    name: round(split_plan["desired_target_boxes_by_class"][class_name][name], 2)
                    for name in SPLIT_NAMES
                },
                "assignment_targets": {
                    name: round(split_plan["assignment_target_boxes_by_class"][class_name][name], 2)
                    for name in SPLIT_NAMES
                },
                "actual_sample_assignment": {
                    name: int(split_plan["actual_target_boxes_by_class"][class_name][name])
                    for name in SPLIT_NAMES
                },
            }
            for class_name in target_class_names
        }
        summary["target_box_planning"][
            "classes_exceeding_test_target_due_to_forced_samples"
        ] = split_plan.get("classes_exceeding_test_target_due_to_forced_samples", [])

    for split_name in SPLIT_NAMES:
        total_boxes, per_class_boxes = count_target_by_class(kept_ids_per_split[split_name])
        summary["counts"][split_name] = {
            "n_samples": len(split[split_name]),
            "n_images": len(kept_ids_per_split[split_name]),
            "n_target_boxes": total_boxes,
            "n_target_boxes_by_class": per_class_boxes,
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
    if split_plan.get("classes_exceeding_test_target_due_to_forced_samples"):
        print(
            "[WARN] Forced test samples already exceed the requested test ratio for: "
            + ", ".join(split_plan["classes_exceeding_test_target_due_to_forced_samples"])
        )
    print(f"[OK] Wrote stationary OVR dataset: {out_dir}")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
