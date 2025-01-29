import shutil

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import os

def extractCandidatePatches(source_folder, destination_folder):

    sub_folders = [f.path for f in os.scandir(source_folder) if f.is_dir()] #since each image we need is in a subfolder

    image_files = []
    for sub_folder in sub_folders:
        files = [f for f in os.listdir(sub_folder) if f.endswith('.png') or f.endswith('.tif')]
        if files:
            files_path = os.path.join(sub_folder, files[0])
            img = Image.open(files_path)
            image_files.append(img)

    for i in range(len(source_folder)):

        # Convert to grayscale for intensity analysis
        gray_image = image_files[i].convert("L")
        gray_array = np.array(gray_image)

        # Display the original and grayscale image
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image_files[i])
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
        low_res_shape = image_files[i].size #(8192, 5641) #Pixel Resolution of loaded image
        high_res_shape = (86306, 59435) #Pixel resolution for full resolution image

        # Correct positive scaling factor
        positive_scaling_factor_x = high_res_shape[0] / low_res_shape[0]  # ≈ 10.5354
        positive_scaling_factor_y = high_res_shape[1] / low_res_shape[1]  # ≈ 10.5375

        tile_size = (2584, 1936)  # High-resolution tile size

        "establish high res grid"
        num_patches_x = high_res_shape[0] // tile_size[0]  # Round up
        num_patches_y = high_res_shape[1] // tile_size[1]  # Round up

        # Generate the coordinate grid
        x_coords = np.arange(0, num_patches_x * tile_size[0], tile_size[0])
        y_coords = np.arange(0, num_patches_y * tile_size[1], tile_size[1])

        # Create a grid of coordinates
        high_res_grid = [(x, y) for y in y_coords for x in x_coords]


        # Calculate exact low-res patch size to perfectly scale to tile size
        patch_size_x = tile_size[0] / positive_scaling_factor_x
        patch_size_y = tile_size[1] / positive_scaling_factor_y

        # Rounding in order to select patches (cannot select non-int pixel values)
        patch_size = [round(patch_size_x), round(patch_size_y)]
        patch_size_unrounded = [patch_size_x, patch_size_y]

        # Function to calculate adaptive thresholds based on image histogram
        def calculate_adaptive_thresholds(image_array):
            percentile_low = np.percentile(image_array, 15)
            percentile_high = np.percentile(image_array, 60)
            dynamic_ratio = 0.20  # Default ratio but can be tuned based on std deviation
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
        start_coordinates = []

        for y in range(0, low_res_shape[1] - patch_size[1] + 1, patch_size[1]):
            for x in range(0, low_res_shape[0] - patch_size[0] + 1, patch_size[0]):
                patch = gray_array[y:y + patch_size[1], x:x + patch_size[0]]
                saturation_status = is_saturated(patch, low_thresh_img, high_thresh_img, ratio_img)
                patch_number = patch_number + 1
                start_coordinates.append((x, y))
                if saturation_status == "balanced":
                    selected_patches.append((x, y))
                    #Saving the patch number for the balanced patches, so we can select them from the high res image folder later
                    patch_number_saved.append(patch_number)

        # Visualization
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(image_files[i])
        for (x, y) in selected_patches:
            rect = patches.Rectangle((x, y), int(patch_size[0]), int(patch_size[1]), linewidth=1, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
        ax.set_title("Selected Balanced Patches Highlighted in Green")
        ax.axis("off")
        plt.show()

        len(selected_patches)  # Number of selected balanced patches

        ''' Randomly select patches that have been deemed ok to analyse '''

        random.seed(42)  # For reproducibility
        selected_random_patchnumbers = random.sample(patch_number_saved, 20)

        selected_random_patches = np.array(start_coordinates)[np.array(selected_random_patchnumbers)]

        "Use selected indices to extract patches via their inherent name"

        patch_image_path = os.path.join(sub_folders[i], os.path.basename(sub_folders[i]) + "_patches")

        # Extract all .tif file names
        tif_files = [f for f in os.listdir(patch_image_path) if f.endswith('.tif')]

        # Filter files based on indices at the specific spot in the name
        fullnames_of_randomly_selected_files = []
        for f in tif_files:
            if 'm' in f:
                last_m_index = f.rfind('m') #creating an index for the last 'm' so we can split using that
                after_m = f[last_m_index + 1:] #find everything after the index we just found (this is the patch number)
                after_m_noExt = os.path.splitext(after_m)[0]

                if after_m_noExt and int(after_m_noExt) in selected_random_patchnumbers:
                    fullnames_of_randomly_selected_files.append(f)

        "move the selected files into a new folder (this folder contains all images to be used in CVAT for annotation"

        for file_name in fullnames_of_randomly_selected_files:
            source_file = os.path.join(patch_image_path, file_name)
            destination_file = os.path.join(destination_folder, file_name)

            shutil.copy(source_file, destination_file)


#

source_folder = "D:\PHD\PhdData\FullSizeSamples"
destination_folder = "D:\PHD\PhdData\DataForAnnotation/"

extractCandidatePatches(source_folder, destination_folder)



