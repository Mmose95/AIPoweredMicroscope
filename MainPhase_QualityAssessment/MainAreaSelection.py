from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the image
#image_path = "E:/PhdData/FullSizeSamples/113331394423/113331394423.png"
#image_path = "E:\PhdData\FullSizeSamples/113101972554/113101972554.tif"
image_path = "E:\PhdData\FullSizeSamples/113331239355/113331239355.tif"
image = Image.open(image_path)

# Convert to grayscale for intensity analysis
gray_image = image.convert("L")
gray_array = np.array(gray_image)

# Display the original and grayscale image
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(gray_array, cmap='gray')
ax[1].set_title("Grayscale Image")
ax[1].axis("off")

plt.tight_layout()
plt.show()

gray_array.shape  # Displaying the image shape for further processing decisions

# Parameters for patching

#calculate resolution indifference (for patching correctly)
low_res_shape = image.size #(8192, 5641) #Pixel Resolution of loaded image
high_res_shape = (86306, 59435) #Pixel resolution for full resolution image

# Correct positive scaling factor
positive_scaling_factor_x = high_res_shape[0] / low_res_shape[0]  # ≈ 10.5354
positive_scaling_factor_y = high_res_shape[1] / low_res_shape[1]  # ≈ 10.5375

tile_size = (2584, 1936)  # High-resolution tile size

# Calculate exact low-res patch size to perfectly scale to tile size
patch_size_x = tile_size[0] / positive_scaling_factor_x
patch_size_y = tile_size[1] / positive_scaling_factor_y

# Rounding in order to select patches (cannot select non-int pixel values)
patch_size = [round(patch_size_x), round(patch_size_y)]
patch_size_unrounded = [patch_size_x, patch_size_y]

# Function to calculate adaptive thresholds based on image histogram
def calculate_adaptive_thresholds(image_array):
    percentile_low = np.percentile(image_array, 5)
    percentile_high = np.percentile(image_array, 70)
    dynamic_ratio = 0.15  # Default ratio but can be tuned based on std deviation
    return percentile_low, percentile_high, dynamic_ratio

low_thresh_img, high_thresh_img, ratio_img = calculate_adaptive_thresholds(gray_image)

# Function to check saturation in a patch
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


# Analyze patches
height, width = gray_array.shape


width = (width // tile_size[0]) * tile_size[0]
height = (height // tile_size[1]) * tile_size[1]


#Patch Selection (in the low res image)
selected_patches = []
patch_number_saved = []
patch_number = 0

for y in range(0, low_res_shape[1] - patch_size[1] + 1, patch_size[1]):
    for x in range(0, low_res_shape[0] - patch_size[0] + 1, patch_size[0]):
        patch = gray_array[y:y + patch_size[1], x:x + patch_size[0]]
        saturation_status = is_saturated(patch, low_thresh_img, high_thresh_img, ratio_img)
        patch_number = patch_number + 1
        if saturation_status == "balanced":
            selected_patches.append((x, y))
            #Saving the patch number for the balanced patches, so we can select them from the high res image folder later
            patch_number_saved.append(patch_number)

# Visualization
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(image)
for (x, y) in selected_patches:
    rect = patches.Rectangle((x, y), int(patch_size[0]), int(patch_size[1]), linewidth=1, edgecolor='green', facecolor='none')
    ax.add_patch(rect)
ax.set_title("Selected Balanced Patches Highlighted in Green")
ax.axis("off")
plt.show()

len(selected_patches)  # Number of selected balanced patches

''' Randomly select patches that have been deemed ok to analyse '''
import random
from PIL import Image

random.seed(42)  # For reproducibility
selected_random_patches = random.sample(patch_number_saved, 20)

#

