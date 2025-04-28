import os
from Helpers_General.LoadImages import load_images_from_folders_fast
from Helpers_General.PatchAndSave import patchNsave, patchNsaveFast_parallel

### Creating patches for SSL ###
#originalFullSizeImages_path = "D:\PHD\PhdData\FullSizeSamples/"  # This path should point to a folder with all the individual tiles/sub-images generated from the microscope (hence pointing to a "_patches" folder is intended)
originalFullSizeImages_path = "C:/Users/SH37YE/Desktop/FullSizeSamples/SSL_Training/OriginalData"  # This path should point to a folder with all the individual tiles/sub-images generated from the microscope (hence pointing to a "_patches" folder is intended)
savePatches_path = "C:/Users/SH37YE/Desktop/FullSizeSamples/SSL_Training/TrainingPatches"  # Whereever we want the copious amount of small patches that constitute the sub-images


if __name__ == "__main__":
    images, filenames, paths = load_images_from_folders_fast(originalFullSizeImages_path, max_workers=48)
    rejectPercent = patchNsaveFast_parallel(images, 512, 512, 0, savePatches_path, filenames, max_workers=34)