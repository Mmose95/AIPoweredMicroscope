import json
from pathlib import Path
from collections import defaultdict
import random

def generate_dataset_for_linear_probing(ssl_data_path, save_folder, val_split=0.2):
    label_dir = (Path(ssl_data_path).parent) / "Supervised" / "Train" / "labels"
    image_dir = (Path(ssl_data_path).parent) / "Supervised" / "Train" / "images"
    save_folder = Path(save_folder)

    output_train_json = save_folder / "pure_patch_class_map_balanced_train.json"
    output_val_json = save_folder / "pure_patch_class_map_balanced_val.json"

    class_to_patches = defaultdict(list)

    # Step 1: Identify pure-class patches with YOLO annotations
    for label_file in label_dir.glob("*.txt"):
        with open(label_file) as f:
            lines = [line.strip().split() for line in f if line.strip()]
            class_ids = [int(line[0]) for line in lines]

        if class_ids and all(cid == class_ids[0] for cid in class_ids):
            img_name = label_file.with_suffix(".jpg").name
            class_to_patches[class_ids[0]].append((img_name, lines))

    # Step 2: Print counts per class
    print("✅ Pure-class patch counts per class:")
    for class_id, files in class_to_patches.items():
        print(f"Class {class_id}: {len(files)} patches")

    min_count = min(len(patches) for patches in class_to_patches.values())
    print(f"\n📉 Limiting class has {min_count} pure patches.")

    # Step 3: Initialize COCO dicts
    def make_coco_dict():
        return {
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": f"class_{i}", "supercategory": "Object"} for i in range(4)]
        }

    train_dict = make_coco_dict()
    val_dict = make_coco_dict()

    annotation_id = 1
    image_id = 1

    for class_id, files in class_to_patches.items():
        selected = random.sample(files, min_count)
        n_val = int(val_split * len(selected))
        val_samples = selected[:n_val]
        train_samples = selected[n_val:]

        for split, sample_list, coco_dict in [("train", train_samples, train_dict), ("val", val_samples, val_dict)]:
            for fname, annotations in sample_list:
                coco_dict["images"].append({
                    "id": image_id,
                    "file_name": fname,
                    "width": 224,  # replace with your actual width
                    "height": 224
                })

                for anno in annotations:
                    cid, x_center, y_center, w, h = map(float, anno)
                    x = (x_center - w / 2) * 224
                    y = (y_center - h / 2) * 224
                    width, height = w * 224, h * 224

                    coco_dict["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(cid),
                        "bbox": [x, y, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    annotation_id += 1
                image_id += 1


    with open(output_train_json, "w") as f:
        json.dump(train_dict, f, indent=2)
    with open(output_val_json, "w") as f:
        json.dump(val_dict, f, indent=2)

    print(f"\n💾 Saved TRAIN COCO annotations: {len(train_dict['images'])} images → {output_train_json}")
    print(f"💾 Saved VAL COCO annotations:   {len(val_dict['images'])} images → {output_val_json}")
