import shutil
from pathlib import Path

def move_images_and_labels(ids, target_img_dir, target_lbl_dir, image_dir, label_dir):
    Path(target_img_dir).mkdir(parents=True, exist_ok=True)
    Path(target_lbl_dir).mkdir(parents=True, exist_ok=True)

    for img_id in ids:
        img_path = Path(image_dir) / f"{img_id}.jpg"
        label_path = Path(label_dir) / f"{img_id}.txt"

        # Print for debugging
        print(f"🖼️ Checking: {img_path} | Exists: {img_path.exists()}")
        print(f"📄 Checking: {label_path} | Exists: {label_path.exists()}")

        try:
            if img_path.exists():
                shutil.copy(img_path, Path(target_img_dir) / img_path.name)
                print(f"✅ Copied image to {Path(target_img_dir) / img_path.name}")
            else:
                print(f"⚠️ Image NOT found: {img_path}")

            if label_path.exists():
                shutil.copy(label_path, Path(target_lbl_dir) / label_path.name)
                print(f"✅ Copied label to {Path(target_lbl_dir) / label_path.name}")
            else:
                print(f"⚠️ Label NOT found: {label_path}")
        except Exception as e:
            print(f"❌ Error copying {img_id}: {e}")

def convert_txt_to_json(image_dir, label_dir):
    import os
    import json
    from PIL import Image
    from glob import glob

    # === Config ===

    from pathlib import Path
    base_path = image_dir.parent

    output_json = base_path / "images/" / "_annotations.coco.json"
    output_json1 = base_path / "_annotations.coco.json"
    image_exts = [".jpg", ".png"]  # Add more if needed
    categories = [
        {"id": 0, "name": "class_0", "supercategory": "Object"},
        {"id": 1, "name": "class_1", "supercategory": "Object"},
        {"id": 2, "name": "class_2", "supercategory": "Object"},
        {"id": 3, "name": "class_3", "supercategory": "Object"},
        # Expand as needed based on your dataset
    ]

    # === Initialize COCO structure ===
    coco = {
        "images": [],
        "annotations": [],
        "categories": categories,
        "supercategories": "Object"
    }

    def yolo_to_coco_bbox(yolo_bbox, img_w, img_h):
        x_center, y_center, w, h = yolo_bbox
        x_center *= img_w
        y_center *= img_h
        w *= img_w
        h *= img_h
        x_min = x_center - w / 2
        y_min = y_center - h / 2
        return [x_min, y_min, w, h]

    # === Main Loop ===
    annotation_id = 0
    image_id = 0

    label_files = sorted(glob(os.path.join(label_dir, "*.txt")))

    for label_file in label_files:
        basename = os.path.splitext(os.path.basename(label_file))[0]

        # Try matching image file
        image_path = None
        for ext in image_exts:
            candidate = os.path.join(image_dir, basename + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if not image_path:
            print(f"⚠️ No image found for {basename}, skipping...")
            continue

        # Load image info
        with Image.open(image_path) as img:
            width, height = img.size

        coco["images"].append({
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        })

        # Parse YOLO annotations
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                yolo_bbox = list(map(float, parts[1:]))
                coco_bbox = yolo_to_coco_bbox(yolo_bbox, width, height)

                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": coco_bbox,
                    "area": coco_bbox[2] * coco_bbox[3],
                    "iscrowd": 0
                })
                annotation_id += 1

        image_id += 1

    # === Save COCO JSON ===
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)
    with open(output_json1, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"✅ COCO-format annotations saved to: {output_json}")


def fullDataSplits(all_img_dir, label_dir, basepath):

    image_dir_Path = Path(all_img_dir)

    all_images = sorted([f.stem for f in image_dir_Path.glob("*.jpg")])  # No extensions
    import random
    random.shuffle(all_images)

    ssl_split_ratio = 0.7
    n_ssl = int(len(all_images) * ssl_split_ratio)

    ssl_image_ids = all_images[:n_ssl]
    supervised_image_ids = all_images[n_ssl:]

    from sklearn.model_selection import train_test_split

    # Split supervised set into train, val, and test (70/15/15 of the 30% chunk)
    train_ids, temp_ids = train_test_split(supervised_image_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    print(f"Total number of images for this run = " + str(len(all_images)))
    print(f"🔧 SSL images: {len(ssl_image_ids)}")
    print(f"✅ Supervised Train: {len(train_ids)}")
    print(f"🧪 Supervised Val: {len(val_ids)}")
    print(f"🧪 Supervised Test: {len(test_ids)}")

    base = Path(basepath)

    #SSL DATA
    move_images_and_labels(ssl_image_ids, base / "SSL/images", base / "SSL/labels", all_img_dir, label_dir)

    #Supervised Training
    move_images_and_labels(train_ids, base / "Supervised/Train/images", base / "Supervised/Train/labels", all_img_dir, label_dir)
    convert_txt_to_json(base / "Supervised/Train/images", base / "Supervised/Train/labels")

    #Supervised Validation
    move_images_and_labels(val_ids, base / "Supervised/Valid/images", base / "Supervised/Valid/labels", all_img_dir, label_dir)
    convert_txt_to_json(base / "Supervised/Valid/images", base / "Supervised/Valid/labels")

    #Supervised Test
    move_images_and_labels(test_ids, base / "Supervised/Test/images", base / "Supervised/Test/labels", all_img_dir, label_dir)
    convert_txt_to_json(base / "Supervised/Test/images", base / "Supervised/Test/labels")


