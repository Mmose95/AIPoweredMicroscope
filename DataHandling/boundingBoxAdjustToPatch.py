import os
import random
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

full_image_annotations_path = "./instances_default.json"

# Directory where patches are stored
patches_dir = "./patchesClean/"
output_annotations_dir = "X:/PhdData/InputPatches_Hvidovre/"
output_annotations_path = "./coco_patched_annotations.json"

# Patch size
PATCH_SIZE = 256

# Load full image annotations
with open(full_image_annotations_path, "r") as f:
    annotations_dicts = json.load(f)

# Extract relevant parts of the data
annotations_list = annotations_dicts["annotations"]
images_info = annotations_dicts["images"]
categories = annotations_dicts["categories"]

# Initialize COCO output structure
coco_output = {
    "licenses": annotations_dicts.get("licenses", []),
    "info": annotations_dicts.get("info", {}),
    "categories": categories,
    "images": [],
    "annotations": []
}

# Create output directory for patched annotations
os.makedirs(output_annotations_dir, exist_ok=True)

# Function to adjust bounding boxes
def adjust_bboxes_for_patch(bboxes, patch_x, patch_y, patch_size):
    adjusted_bboxes = []
    for bbox in bboxes:
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height

        # Check if the bounding box overlaps with the patch
        if x_max > patch_x and x_min < patch_x + patch_size and y_max > patch_y and y_min < patch_y + patch_size:
            # Calculate new bounding box within the patch
            new_x_min = max(0, x_min - patch_x)
            new_y_min = max(0, y_min - patch_y)
            new_x_max = min(patch_size, x_max - patch_x)
            new_y_max = min(patch_size, y_max - patch_y)

            # Only include boxes that remain valid after cropping
            if new_x_min < new_x_max and new_y_min < new_y_max:
                adjusted_bboxes.append([new_x_min, new_y_min, new_x_max - new_x_min, new_y_max - new_y_min])
    return adjusted_bboxes

# Unique ID counters
annotation_id = 1
patch_image_id = 1

# Process each image and its annotations
for image_info in images_info:
    image_id = image_info["id"]
    img_width = image_info["width"]
    img_height = image_info["height"]

    # Get all bounding boxes for this image
    bboxes = [
        annotation["bbox"]
        for annotation in annotations_list
        if annotation["image_id"] == image_id
    ]
    categories_per_bbox = [
        annotation["category_id"]
        for annotation in annotations_list
        if annotation["image_id"] == image_id
    ]

    # Loop through patches of the image
    for patch_x in range(0, img_width, PATCH_SIZE):
        for patch_y in range(0, img_height, PATCH_SIZE):
            # Add patch as a new image
            patch_file_name = f"{image_info['file_name']}_patch_{patch_x}_{patch_y}.png"
            coco_output["images"].append({
                "id": patch_image_id,
                "width": PATCH_SIZE,
                "height": PATCH_SIZE,
                "file_name": patch_file_name,
                "license": image_info.get("license", 0),
                "flickr_url": image_info.get("flickr_url", ""),
                "coco_url": image_info.get("coco_url", ""),
                "date_captured": image_info.get("date_captured", "")
            })

            # Adjust bounding boxes for the patch
            adjusted_bboxes = adjust_bboxes_for_patch(bboxes, patch_x, patch_y, PATCH_SIZE)

            # Add annotations for this patch
            for bbox, category_id in zip(adjusted_bboxes, categories_per_bbox):
                x_min, y_min, width, height = bbox
                area = width * height
                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": patch_image_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, width, height],
                    "area": area,
                    "segmentation": [],
                    "iscrowd": 0
                })
                annotation_id += 1

            patch_image_id += 1

# Save the COCO-format output
with open(output_annotations_path, "w") as f:
    json.dump(coco_output, f, indent=4)

