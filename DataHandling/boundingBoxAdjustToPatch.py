import json
import os

full_image_annotations_path = "./instances_default.json"

# Directory where patches are stored
patches_dir = "./patchesClean/"
output_annotations_dir = "./DataHandling/LabeledPatches/"

# Patch size
PATCH_SIZE = 256

# Load full image annotations
with open(full_image_annotations_path, "r") as f:
    annotations_dicts = json.load(f)

# Extract relevant parts of the data
annotations_list = annotations_dicts["annotations"]
images_info = annotations_dicts["images"]

# Create output directory for patched annotations
os.makedirs(output_annotations_dir, exist_ok=True)

def adjust_bboxes_for_patch(bboxes, patch_x, patch_y, patch_size):
    adjusted_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox

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

# Group annotations by image ID
grouped_annotations = {}
for annotation in annotations_list:
    image_id = annotation["image_id"]
    if image_id not in grouped_annotations:
        grouped_annotations[image_id] = []
    x_min, y_min, width, height = annotation["bbox"]
    x_max = x_min + width
    y_max = y_min + height
    grouped_annotations[image_id].append([x_min, y_min, x_max, y_max])

# Iterate through images and their annotations
for image_id, bboxes in grouped_annotations.items():
    # Get image dimensions
    img_info = next(img for img in images_info if img["id"] == image_id)
    img_width = img_info["width"]
    img_height = img_info["height"]

    # Loop through all patches of this image
    for patch_x in range(0, img_width, PATCH_SIZE):
        for patch_y in range(0, img_height, PATCH_SIZE):
            # Calculate patch ID
            patch_id = f"{image_id}_patch_{patch_x}_{patch_y}"

            # Adjust bounding boxes for this patch
            adjusted_bboxes = adjust_bboxes_for_patch(bboxes, patch_x, patch_y, PATCH_SIZE)

            # Save patch annotations if there are any bounding boxes
            if adjusted_bboxes:
                patch_annotation = {
                    "image_id": patch_id,
                    "width": PATCH_SIZE,
                    "height": PATCH_SIZE,
                    "bboxes": adjusted_bboxes
                }
                output_path = os.path.join(output_annotations_dir, f"{patch_id}.json")
                with open(output_path, "w") as f:
                    json.dump(patch_annotation, f, indent=4)