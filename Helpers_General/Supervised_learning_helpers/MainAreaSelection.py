""" Function of the script:
Load all patches of a full sample and do a very broad candidate area selection either to be used for: 1) pull out areas that are realistic areas
(in terms of what the microbiologist would also choose to look at) to generate examples of objects to train on in a supervised manner.
2) Can be used as a pre-analysis/efficiency-analysis in the final product with the purpose of not having to analyse all individual images but rather just until satisfied
"""

import shutil
import random
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
import re
from PIL import Image
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

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

def load_tile(tile_file, patch_folder):
    tile_path = os.path.join(patch_folder, tile_file)
    img = Image.open(tile_path)
    tile_img = img.convert("L")
    tile_img_color = img.convert("RGB")
    return np.array(tile_img), np.array(tile_img_color), tile_path

def extractCandidatePatches(source_folder, destination_folder, downscale_factor, tile_size):
    """
    Processes high-resolution images by stitching tiles together.
    Assumes that the files in the folder are arranged in a meander style.
    """
    sub_folders = [f.path for f in os.scandir(source_folder) if f.is_dir()]  # Get subfolders

    for sub_folder in sub_folders:
        print(f"Processing folder: {sub_folder}: folder index: {sub_folders.index(sub_folder)+1} / {len(sub_folders)}")

        all_tile_arrays = []
        all_tile_arrays_RGB = []
        tile_paths = []

        # Load image tiles from the subfolder
        patch_folder = os.path.join(sub_folder, os.path.basename(sub_folder) + "_patches")
        tile_files = [f for f in os.listdir(patch_folder) if f.endswith('.png') or f.endswith('.tif')]
        # Try using natural sort to ensure proper ordering:
        tile_files.sort(key=natural_key)

        #print("Tile files order:")
        #for i, f_name in enumerate(tile_files):
            #print(f"{i}: {f_name}")

        start_time = time.time()
        #loading images in parallel to speed up the process! (load_tile is the loader function)
        bound_loader = partial(load_tile, patch_folder=patch_folder)
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(bound_loader, tile_files))

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"\nLoaded {len(tile_files)} images in {elapsed:.2f} seconds.")

        # Unpack results
        for tile_img, tile_img_color, tile_path in results:
            all_tile_arrays.append(tile_img)
            all_tile_arrays_RGB.append(tile_img_color)
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
        if actual_tile_count != expected_tile_count:
            print("Warning: Number of tiles does not match grid dimensions!")
        else:
            print(f"Expected tile count: {expected_tile_count}, actual tile count: {actual_tile_count} --> Tiles match!")

        # Create a blank image for stitching
        stitched_width = num_cols * tile_size[0]
        stitched_height = num_rows * tile_size[1]
        stitched_image = Image.new("RGB", (stitched_width, stitched_height))

        #Map the tiles into the grid
        for k, tile_array in enumerate(all_tile_arrays_RGB):
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


        selected_regions = []  # Store tuples of (top-left index, [4 paths])
        for col in range(0, num_cols - 1, 2):
            for row in range(0, num_rows - 1, 2):
                idx_tl = row + col * num_rows
                idx_tr = (row + 1) + col * num_rows
                idx_bl = row + (col) * num_rows + num_rows
                idx_br = (row + 1) + (col) * num_rows + num_rows

                if max(idx_tl, idx_tr, idx_bl, idx_br) >= len(all_tile_arrays):
                    continue

                # Build 2x2 patch
                top_row = np.hstack([all_tile_arrays[idx_tl], all_tile_arrays[idx_tr]])
                bottom_row = np.hstack([all_tile_arrays[idx_bl], all_tile_arrays[idx_br]])
                combined_tile = np.vstack([top_row, bottom_row])

                saturation_status = is_saturated(combined_tile, global_low_thresh, global_high_thresh, global_ratio)
                if saturation_status == "balanced":
                    selected_regions.append((
                        idx_tl,  # top-left index
                        [tile_paths[idx_tl], tile_paths[idx_tr], tile_paths[idx_bl], tile_paths[idx_br]]
                    ))

        print(f"Total selected 2x2 regions: {len(selected_regions)}")

        # Randomly pick from the 2x2 tile regions
        num_to_sample = min(20, len(selected_regions))
        sampled_regions = random.sample(selected_regions, num_to_sample)

        for i, (idx_tl, paths) in enumerate(sampled_regions):
            idx_tr = idx_tl + 1
            idx_bl = idx_tl + num_rows
            idx_br = idx_bl + 1

            # Merge RGB tiles
            top_row = np.hstack([all_tile_arrays_RGB[idx_tl], all_tile_arrays_RGB[idx_bl]])
            bottom_row = np.hstack([all_tile_arrays_RGB[idx_tr], all_tile_arrays_RGB[idx_br]])
            merged_image = np.vstack([top_row, bottom_row])

            # Convert and save
            merged_pil = Image.fromarray(merged_image)
            sample_name = os.path.basename(tile_paths[0]).split("_")[0]
            save_name = f"{sample_name}region_{i:02d}.png"
            save_path = os.path.join(destination_folder, save_name)
            merged_pil.save(save_path)

        # ---- Visualization ----
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(stitched_image_resized)
        #plotting non-selected regions
        for idx_tl, _ in selected_regions:
            col = idx_tl // num_rows
            row = idx_tl % num_rows
            x_scaled = int(col * tile_size[0] * downscale_factor)
            y_scaled = int(row * tile_size[1] * downscale_factor)
            w_scaled = int(tile_size[0] * 2 * downscale_factor)
            h_scaled = int(tile_size[1] * 2 * downscale_factor)
            rect = patches.Rectangle((x_scaled, y_scaled), w_scaled, h_scaled,
                                     linewidth=1, edgecolor="red", facecolor="none")
            ax.add_patch(rect)
        #plotting selected regions
        for idx_tl, _ in sampled_regions:
            col = idx_tl // num_rows
            row = idx_tl % num_rows
            x_scaled = int(col * tile_size[0] * downscale_factor)
            y_scaled = int(row * tile_size[1] * downscale_factor)
            w_scaled = int(tile_size[0] * 2 * downscale_factor)
            h_scaled = int(tile_size[1] * 2 * downscale_factor)
            rect = patches.Rectangle((x_scaled, y_scaled), w_scaled, h_scaled,
                                     linewidth=2, edgecolor="green", facecolor="none")
            ax.add_patch(rect)

        sample_name = os.path.basename(tile_paths[0]).split("_")[0]
        plt.title(f"sample {sample_name}_Candidate Overview.png")
        plt.axis("off")
        overview_path = os.path.join(destination_folder.split("SelectedFOVs")[0] + "/CandidateOverview/",
                                     f"sample {sample_name}_Candidate Overview.png")
        os.makedirs(os.path.dirname(overview_path), exist_ok=True)
        plt.savefig(overview_path)
        plt.show()

# **Execution**
source_folder = "C:/Users/SH37YE/Desktop/FullSizeSamples/QA_Supervised/QA_Supervised_TestData/Original/" #Source folder should be the folder containing everything for a digitalized image i.e. should be just a path + sample number
destination_folder = "C:/Users/SH37YE/Desktop/FullSizeSamples/QA_Supervised/QA_Supervised_TestData\SelectedFOVs/"
extractCandidatePatches(source_folder, destination_folder, downscale_factor=0.05, tile_size = (2584, 1936))
