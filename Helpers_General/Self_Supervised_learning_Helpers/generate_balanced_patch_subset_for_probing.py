import json
from pathlib import Path
from collections import defaultdict
import random


'''This code generates a subset of data from the intended supervised training data used to train a linear classifier ontop of the SSL model from DINOV2
which is used to determine which model (encoder) will be used as starting point in the downstream task'''

def generate_dataset_for_linear_probing(ssl_data_path, save_folder):

    #SSL_data_path point to the full data for supervised, here we direct it to the .txt files for the origianl data.
    #wierd way of doing it but ok..
    label_dir = (Path(ssl_data_path).parent) / "Supervised" / "Train" / "labels"

    # Track pure-class patches per class
    class_to_patches = defaultdict(list)

    # Step 1: Identify pure-class patches
    for label_file in label_dir.glob("*.txt"):
        with open(label_file) as f:
            class_ids = [int(line.strip().split()[0]) for line in f if line.strip()]

        if class_ids and all(cid == class_ids[0] for cid in class_ids):
            img_name = label_file.with_suffix(".jpg").name
            class_to_patches[class_ids[0]].append(img_name)

    # Step 2: Get class counts
    print("✅ Pure-class patch counts per class:")
    for class_id, files in class_to_patches.items():
        print(f"Class {class_id}: {len(files)} patches")

    # Step 3: Find the limiting class
    min_count = min(len(patches) for patches in class_to_patches.values())
    print(f"\n📉 Limiting class has {min_count} pure patches. Will sample this many from each class.")

    # Step 4: Build balanced subset
    balanced_map = {}
    for class_id, files in class_to_patches.items():
        selected = random.sample(files, min_count)
        for fname in selected:
            balanced_map[fname] = class_id

    # Save to JSON
    output_json = Path(save_folder) / "pure_patch_class_map_balanced.json"
    with open(output_json, "w") as f:
        json.dump(balanced_map, f, indent=2)

    print(f"\n💾 Saved balanced patch map with {len(balanced_map)} total entries to:\n{output_json}")
