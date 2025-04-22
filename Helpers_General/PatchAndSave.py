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


# Helper to check entropy on ndarray
def is_low_entropy_ndarray(patch_array, threshold=3.8):
    if patch_array.ndim == 3 and patch_array.shape[2] == 3:
        gray = rgb2gray(patch_array)
    else:
        gray = patch_array
    entropy = shannon_entropy(gray)
    return entropy < threshold

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

            if is_low_entropy_ndarray(patch):
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

    sample_name = fname.split("_")[0]
    rejected_dir = savePath / f"{sample_name}_rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)

    accepted = 0
    rejected = 0

    for i, j in grid:
        patch_array = img_array[i:i + d, j:j + e]
        if patch_array.shape[0] != d or patch_array.shape[1] != e:
            continue

        name = f"{Path(fname).stem}_patch_{i}_{j}.png"
        if is_low_entropy_ndarray(patch_array):
            if random.random() <= 0.05:
                Image.fromarray(patch_array).save(rejected_dir / name)
                rejected += 1
            continue

        Image.fromarray(patch_array).save(savePath / name)
        accepted += 1

    return accepted, rejected

# Main parallel patching function
def patchNsaveFast_parallel(images, d, e, overlapPercent, savePath, filenames, max_workers):
    savePath = Path(savePath)
    savePath.mkdir(parents=True, exist_ok=True)
    overlapPix = int(overlapPercent / 100 * d) if overlapPercent > 0 else d

    results = Parallel(n_jobs=max_workers)(
        delayed(process_image_patches)(img, fname, d, e, overlapPix, savePath)
        for img, fname in zip(images, filenames)
    )

    total_accepted = sum(r[0] for r in results)
    total_rejected = sum(r[1] for r in results)

    print(f"✅ Kept: {total_accepted} patches")
    print(f"🗑️ Rejected (saved 5%): {total_rejected} patches")