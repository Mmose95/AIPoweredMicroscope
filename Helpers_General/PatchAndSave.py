""" The purpose of this code is to take large original images and convert them to patches manageable for processing in the following NN (and saving them to a specified folder
 This code is independent of the main and should not be run each time a new training is done. Instead, use the DataLoader to load patches directly. Note: "Dataloader" is implemented directly into each individual training loop"""

from pathlib import Path
from itertools import product
import numpy as np
from skimage.measure import shannon_entropy
from skimage.color import rgb2gray
import random
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm
import cv2



# Helper to check entropy on ndarray
def is_low_entropy_ndarray(patch_array, threshold=3.8):
    if patch_array.ndim == 3 and patch_array.shape[2] == 3:
        gray = rgb2gray(patch_array)
    else:
        gray = patch_array
    entropy = shannon_entropy(gray)
    return entropy < threshold

# 'Manual' approach where colors of a normal gram stain is defined and patches are checked against this using a predefined percentage (of the patch which must match)
def is_gram_stained_patch(patch_array, color_thresh=0.05):
    """
    Returns True if a patch contains 'enough' Gram-stain coloration.
    Uses HSV color space and masks for typical purple/pink/blue hues.
    """
    if patch_array.ndim == 3 and patch_array.shape[2] == 3:
        hsv = cv2.cvtColor(patch_array, cv2.COLOR_RGB2HSV)

        # Define HSV color range for Gram stain (purple/pink/blue)
        masks = []
        # Purple hues (Gram+)
        masks.append(cv2.inRange(hsv,(125, 50, 50), (155, 255, 255)))
        # Pinkish-red (Gram-)
        masks.append(cv2.inRange(hsv,(160, 50, 50), (180, 255, 255)))
        # Light blue background sometimes seen
        masks.append(cv2.inRange(hsv,(90, 30, 30), (130, 255, 255)))

        full_mask = sum(masks)
        colored_ratio = np.count_nonzero(full_mask) / full_mask.size

        return colored_ratio <= color_thresh  # e.g., 2% pixels must match
    return False

def patchNsave(images, d, e, overlapPercent, savePath, samples):
    patches_nd = []
    rejected_patches = []

    # Create subfolders for accepted and rejected patches
    savePath = Path(savePath)
    accepted_dir = savePath
    rejected_dir = savePath / Path(samples + "_rejected")
    #accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    if overlapPercent > 0:
        overlapPix = int(overlapPercent / 100 * d)
    else:
        overlapPix = d

    for image in images:
        w, h = image.size
        grid = list(product(range(0, h - h % d, overlapPix), range(0, w - w % e, overlapPix)))

        for i, j in grid:
            box = (j, i, j + d, i + e)
            patch = image.crop(box)
            name = f"{Path(image.filename).stem}_patch_{i}_{j}.png"

            if is_gram_stained_patch(patch):
                # Save 10% of rejected patches randomly for validation
                if random.random() <= 0.05:
                    patch.save(rejected_dir / name)
                    rejected_patches.append(np.array(patch))
                continue  # Skip adding this patch to the accepted list

            # Save accepted patch
            patch.save(accepted_dir / name)
            patches_nd.append(np.array(patch))

    print(f"✅ Kept: {len(patches_nd)} patches")
    print(f"🗑️ Rejected (based on shannon_entropy > 5): {len(rejected_patches)} patches")

# Function to process and save patches from one image
def process_image_patches(img_array, fname, d, e, overlapPix, savePath):
    h, w = img_array.shape[:2]
    grid = list(product(range(0, h - h % d, overlapPix), range(0, w - w % e, overlapPix)))

    sample_name = Path(fname).stem.split("_")[0]
    rejected_dir = savePath / f"{sample_name}_rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)

    accepted = 0
    rejected = 0

    for i, j in grid:
        patch_array = img_array[i:i + d, j:j + e]
        if patch_array.shape[0] != d or patch_array.shape[1] != e:
            continue

        name = f"{Path(fname).stem}_patch_{i}_{j}.png"
        if is_gram_stained_patch(patch_array):
            if random.random() <= 0.10: #Lets save 10% of rejected patches to include some background!
                Image.fromarray(patch_array).save(rejected_dir / name)
                rejected += 1
            continue

        Image.fromarray(patch_array).save(savePath / name)
        accepted += 1

    return accepted, rejected

# Main parallel patching function
def patchNsaveFast_parallel(images, d, e, overlapPercent, savePath, filenames, max_workers):
    assert len(images) == len(filenames), "Mismatch between images and filenames" #checking that we have the same images as filenames (otherwise we have a bug)
    savePath = Path(savePath)
    savePath.mkdir(parents=True, exist_ok=True)
    overlapPix = int(overlapPercent / 100 * d) if overlapPercent > 0 else d

    results = Parallel(n_jobs=max_workers)(
        delayed(process_image_patches)(img, fname, d, e, overlapPix, savePath)
        for img, fname in tqdm(list(zip(images, filenames)))
    )

    total_accepted = sum(r[0] for r in results)
    total_rejected = sum(r[1] for r in results)

    print(f"✅ Kept: {total_accepted} patches")
    print(f"🗑️ Rejected (based on gram-stain color match): {total_rejected} patches")
    rejectPercent = (total_rejected / (total_rejected + total_accepted)) * 100
    print(f" Reject percent: {rejectPercent:.2f}%")
    return rejectPercent