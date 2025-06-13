import os
import json
import shutil
from pathlib import Path
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
import random

def move_images_and_labels(ids, target_img_dir, target_lbl_dir, image_dir, label_dir):
    Path(target_img_dir).mkdir(parents=True, exist_ok=True)
    Path(target_lbl_dir).mkdir(parents=True, exist_ok=True)
    for img_id in ids:
        img_path = Path(image_dir) / f"{img_id}.jpg"
        label_path = Path(label_dir) / f"{img_id}.txt"
        if img_path.exists():
            shutil.copy(img_path, Path(target_img_dir) / img_path.name)
        if label_path.exists():
            shutil.copy(label_path, Path(target_lbl_dir) / label_path.name)

def convert_txt_to_json(image_dir, label_dir):
    output_json = Path(image_dir).parent / "_annotations.coco.json"
    image_exts = [".jpg", ".png"]
    categories = [{"id": i, "name": f"class_{i}", "supercategory": "Object"} for i in range(4)]

    coco = {"images": [], "annotations": [], "categories": categories}
    annotation_id = 0
    image_id = 0

    label_files = sorted(glob(str(label_dir / "*.txt")))

    for label_file in label_files:
        basename = Path(label_file).stem
        image_path = next((str(image_dir / f"{basename}{ext}") for ext in image_exts if (image_dir / f"{basename}{ext}").exists()), None)
        if not image_path:
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        coco["images"].append({
            "id": image_id,
            "file_name": Path(image_path).name,
            "width": width,
            "height": height
        })

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:])
                x_min = (x_center - w / 2) * width
                y_min = (y_center - h / 2) * height
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, w * width, h * height],
                    "area": w * h * width * height,
                    "iscrowd": 0
                })
                annotation_id += 1
        image_id += 1

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"✅ COCO-format annotations saved to: {output_json}")

    # Inject 'images/' after 'Train', 'Valid', or 'Test'
    if output_json.name == "_annotations.coco.json":
        if output_json.parent.name in ["Train", "Valid", "Test"]:
            images_folder = output_json.parent / "images"
            images_folder.mkdir(parents=True, exist_ok=True)
            output_json_images = images_folder / "_annotations.coco.json"
            shutil.copy(output_json, output_json_images)
            print(f"📂 Also copied to: {output_json_images}")


from pathlib import Path
from itertools import combinations

def sanity_check_split(name, image_dir, label_dir):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)

    image_ids = set(f.stem for f in image_dir.glob("*.jpg"))
    label_ids = set(f.stem for f in label_dir.glob("*.txt"))

    missing_labels = image_ids - label_ids
    missing_images = label_ids - image_ids
    matched = image_ids & label_ids

    print(f"\n🧪 Sanity check for split: {name}")
    print(f" - Total images: {len(image_ids)}")
    print(f" - Total labels: {len(label_ids)}")
    print(f" - ✅ Matched image-label pairs: {len(matched)}")
    if missing_labels:
        print(f"⚠️ {len(missing_labels)} image(s) missing labels: {list(missing_labels)[:5]}")
    if missing_images:
        print(f"⚠️ {len(missing_images)} label(s) missing images: {list(missing_images)[:5]}")

    return image_ids


def extract_sample_ids(image_ids):
    return set(img_id.split("_")[0] for img_id in image_ids)


def check_sample_overlap(splits_dict):
    print("\n🔍 Checking for sample ID overlap between splits...")
    for (name1, ids1), (name2, ids2) in combinations(splits_dict.items(), 2):
        overlap = ids1 & ids2
        if overlap:
            print(f"❌ Overlap between {name1} and {name2}: {len(overlap)} sample(s) → {sorted(list(overlap))[:5]}")
            raise ValueError(f"Sample overlap detected between {name1} and {name2}. Please fix the splits!!!!!.")
        else:
            print(f"✅ No overlap between {name1} and {name2}")


def run_all_sanity_checks(base_path):
    base = Path(base_path)
    ssl_ids = sanity_check_split("SSL", base / "SSL/images", base / "SSL/labels")
    train_ids = sanity_check_split("Supervised Train", base / "Supervised/Train/images", base / "Supervised/Train/labels")
    val_ids = sanity_check_split("Supervised Valid", base / "Supervised/Valid/images", base / "Supervised/Valid/labels")
    test_ids = sanity_check_split("Supervised Test", base / "Supervised/Test/images", base / "Supervised/Test/labels")

    split_samples = {
        "SSL": extract_sample_ids(ssl_ids),
        "Train": extract_sample_ids(train_ids),
        "Valid": extract_sample_ids(val_ids),
        "Test": extract_sample_ids(test_ids),
    }

    check_sample_overlap(split_samples)


def fullDataSplits_SampleSafe(all_img_dir, label_dir, basepath, ssl_ratio=0.7):
    image_dir = Path(all_img_dir)
    base = Path(basepath)
    sample_to_images = {}

    for img_path in image_dir.glob("*.jpg"):
        sample_id = img_path.stem.split("_")[0]
        sample_to_images.setdefault(sample_id, []).append(img_path.stem)

    all_sample_ids = list(sample_to_images.keys())
    random.seed(42)
    random.shuffle(all_sample_ids)

    n_ssl = int(len(all_sample_ids) * ssl_ratio)
    ssl_samples = all_sample_ids[:n_ssl]
    supervised_samples = all_sample_ids[n_ssl:]

    train_ids, temp_ids = train_test_split(supervised_samples, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    def expand(sample_ids):
        return [img_id for sid in sample_ids for img_id in sample_to_images[sid]]

    ssl_image_ids = expand(ssl_samples)
    train_image_ids = expand(train_ids)
    val_image_ids = expand(val_ids)
    test_image_ids = expand(test_ids)

    # Copy and convert
    move_images_and_labels(ssl_image_ids, base / "SSL/images", base / "SSL/labels", all_img_dir, label_dir)
    convert_txt_to_json(base / "Supervised/Train/images", base / "Supervised/Train/labels")

    move_images_and_labels(train_image_ids, base / "Supervised/Train/images", base / "Supervised/Train/labels", all_img_dir, label_dir)
    convert_txt_to_json(base / "Supervised/Train/images", base / "Supervised/Train/labels")
    move_images_and_labels(val_image_ids, base / "Supervised/Valid/images", base / "Supervised/Valid/labels", all_img_dir, label_dir)
    convert_txt_to_json(base / "Supervised/Valid/images", base / "Supervised/Valid/labels")
    move_images_and_labels(test_image_ids, base / "Supervised/Test/images", base / "Supervised/Test/labels", all_img_dir, label_dir)
    convert_txt_to_json(base / "Supervised/Test/images", base / "Supervised/Test/labels")

    # Summary
    print(f"\n📊 Final Split Summary:")
    print(f"🔧 SSL samples: {len(ssl_samples)} → {len(ssl_image_ids)} images")
    print(f"✅ Train samples: {len(train_ids)} → {len(train_image_ids)} images")
    print(f"🧪 Val samples:   {len(val_ids)} → {len(val_image_ids)} images")
    print(f"🧪 Test samples:  {len(test_ids)} → {len(test_image_ids)} images")

    run_all_sanity_checks(basepath) #SANITY CHECK - across all splits (SSL, Test, Train and Val)