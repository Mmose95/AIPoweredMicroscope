""" Function of the script:
Load all patches of a full sample and do a very broad candidate area selection either to be used for: 1) pull out areas that are realistic areas
(in terms of what the microbiologist would also choose to look at) to generate examples of objects to train on in a supervised manner.
2) Can be used as a pre-analysis/efficiency-analysis in the final product with the purpose of not having to analyse all individual images but rather just until satisfied
"""

import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import xml.etree.ElementTree as ET
import re

def natural_key(string_):
    """Helper function for natural sort"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

# Function to check saturation using **global thresholds**
def is_saturated(patch, low_thresh, high_thresh, ratio):
    total_pixels = patch.size
    low_pixels = np.sum(patch < low_thresh)
    high_pixels = np.sum(patch > high_thresh)
    if (low_pixels / total_pixels) > ratio:
        return "under"
    elif (high_pixels / total_pixels) > ratio:
        return "over"
    else:
        return "balanced"

def calculate_adaptive_thresholds(image_array):
    percentile_low = np.percentile(image_array, 15)
    percentile_high = np.percentile(image_array, 60)
    dynamic_ratio = 0.20  # Default ratio, can be tuned
    return percentile_low, percentile_high, dynamic_ratio

def extractCandidatePatches(source_folder, destination_folder, downscale_factor):
    """
    Processes high-resolution images by stitching tiles together.
    Assumes that the files in the folder are arranged in a meander style.
    """
    sub_folders = [f.path for f in os.scandir(source_folder) if f.is_dir()]  # Get subfolders

    tile_size = (2584, 1936)  # Known tile size

    for sub_folder in sub_folders:
        print(f"Processing folder: {sub_folder}")

        all_tile_arrays = []
        all_tile_arrays_RGB = []
        tile_paths = []

        # Load image tiles from the subfolder
        patch_folder = os.path.join(sub_folder, os.path.basename(sub_folder) + "_patches")
        tile_files = [f for f in os.listdir(patch_folder) if f.endswith('.png') or f.endswith('.tif')]
        # Try using natural sort to ensure proper ordering:
        tile_files.sort(key=natural_key)

        print("Tile files order:")
        for i, f_name in enumerate(tile_files):
            print(f"{i}: {f_name}")

        for idx, tile_file in enumerate(tile_files, start=1):
            tile_path = os.path.join(patch_folder, tile_file)
            tile_img = Image.open(tile_path).convert("L")  # Convert to grayscale
            tile_img_color = Image.open(tile_path)
            print(f"Loading tile {idx} of {len(tile_files)}")
            all_tile_arrays_RGB.append(np.array(tile_img_color))
            all_tile_arrays.append(np.array(tile_img))
            tile_paths.append(tile_path)

        # Extract grid dimensions from the metadata XML file
        xml_path = os.path.join(sub_folder, os.path.basename(sub_folder) + "_metadata.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        tile_region = root.find(".//TileRegions/TileRegion")
        num_cols = int(tile_region.find("Columns").text)
        num_rows = int(tile_region.find("Rows").text)

        print(f"Extracted Grid Size: {num_cols} x {num_rows} from {xml_path}")
        expected_tile_count = num_cols * num_rows
        actual_tile_count = len(all_tile_arrays)
        print(f"Expected tile count: {expected_tile_count}, actual tile count: {actual_tile_count} --> Tiles match!")
        if actual_tile_count != expected_tile_count:
            print("Warning: Number of tiles does not match grid dimensions!")

        # Create a blank image for stitching
        stitched_width = num_cols * tile_size[0]
        stitched_height = num_rows * tile_size[1]
        stitched_image = Image.new("RGB", (stitched_width, stitched_height))

        # Choose the mapping depending on your file ordering.
        for k, tile_array in enumerate(all_tile_arrays_RGB):
            # Mapping (try column-major first):
            col = k // num_rows
            row = k % num_rows

            x_pos = col * tile_size[0]
            y_pos = row * tile_size[1]

            tile_img = Image.fromarray(tile_array)
            stitched_image.paste(tile_img, (x_pos, y_pos))

        # Optional: Draw grid lines and tile indices on the stitched image for debugging
        draw = Image.new("RGB", (stitched_width, stitched_height))
        draw.paste(stitched_image)

        import PIL.ImageDraw as ImageDraw
        import PIL.ImageFont as ImageFont
        draw_obj = ImageDraw.Draw(draw)
        # You might need to adjust the font path depending on your system
        try:
            font = ImageFont.truetype("arial.ttf", 50)
        except:
            font = ImageFont.load_default()

        # Label each tile with its index (or computed grid coordinates)
        for k in range(actual_tile_count):
            # Use the same mapping as above:
            col = k // num_rows
            row = k % num_rows
            x = col * tile_size[0] + 20  # Offset for visibility
            y = row * tile_size[1] + 20
            label = f"{k}\n({col},{row})"
            draw_obj.text((x, y), label, fill="yellow", font=font)

        # Downscale the stitched image for easier visualization (optional)
        stitched_image_resized = draw.resize(
            (int(stitched_image.width * downscale_factor),
             int(stitched_image.height * downscale_factor)),
            Image.Resampling.LANCZOS
        )

        print(f"Full-resolution image size: {stitched_image.size}")
        print(f"Downscaled image size: {stitched_image_resized.size}")

        stitched_gray = stitched_image.convert("L")
        global_low_thresh, global_high_thresh, global_ratio = calculate_adaptive_thresholds(np.array(stitched_gray))
        print(f"Computed Global Thresholds - Low: {global_low_thresh}, High: {global_high_thresh}, Ratio: {global_ratio}")

        selected_patches = []
        patch_number_saved = []
        patch_number = 0

        for x, tile_array in enumerate(all_tile_arrays):
            saturation_status = is_saturated(tile_array, global_low_thresh, global_high_thresh, global_ratio)
            patch_number += 1
            if saturation_status == "balanced":
                selected_patches.append(tile_paths[x])  # Store the corresponding file path
                patch_number_saved.append(patch_number)
        print(f"Total selected patches after global thresholding: {len(selected_patches)}")

        # **Step 4: Randomly Select and Move Patches**
        num_to_sample = min(20, len(selected_patches))
        selected_random_patches = random.sample(selected_patches, num_to_sample)
        print(f"Final selected patches: {selected_random_patches}")

        # Move selected files
        for source_file in selected_random_patches:
            file_name = os.path.basename(source_file)
            destination_file = os.path.join(destination_folder, file_name)
            shutil.copy(source_file, destination_file)

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(stitched_image_resized)

        #compute rectangles to mark all candidate selected patches
        for source_file in selected_patches:
            # Find the tile index from the tile_paths list.
            tile_idx = tile_paths.index(source_file)

            # Compute column and row using the same logic.
            col = tile_idx // num_rows
            row = tile_idx % num_rows
            # Compute scaled coordinates for visualization.
            x_scaled = int(col * tile_size[0] * downscale_factor)
            y_scaled = int(row * tile_size[1] * downscale_factor)
            w_scaled = int(tile_size[0] * downscale_factor)
            h_scaled = int(tile_size[1] * downscale_factor)
            rect = patches.Rectangle((x_scaled, y_scaled), w_scaled, h_scaled,
                                     linewidth=1, edgecolor="red", facecolor="none")
            ax.add_patch(rect)

        #compute rectangles to mark the randomly selected candidate patches
        for source_file in selected_random_patches:
            # Find the tile index from the tile_paths list.
            tile_idx = tile_paths.index(source_file)

            # Compute column and row using the same logic.
            col = tile_idx // num_rows
            row = tile_idx % num_rows
            # Compute scaled coordinates for visualization.
            x_scaled = int(col * tile_size[0] * downscale_factor)
            y_scaled = int(row * tile_size[1] * downscale_factor)
            w_scaled = int(tile_size[0] * downscale_factor)
            h_scaled = int(tile_size[1] * downscale_factor)
            rect = patches.Rectangle((x_scaled, y_scaled), w_scaled, h_scaled,
                                     linewidth=1, edgecolor="green", facecolor="none")
            ax.add_patch(rect)


        plt.title("Stitched Image with Debug Grid and Labels")
        plt.axis("off")
        plt.show()

# **Execution**
source_folder = "D:/PHD/PhdData/FullSizeSamples"
destination_folder = "D:/PHD/PhdData/DataForAnnotation/"
extractCandidatePatches(source_folder, destination_folder, downscale_factor=0.05)
