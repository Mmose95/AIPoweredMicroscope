import os
from Helpers_General.LoadImages import load_images_from_folders_fast
from Helpers_General.PatchAndSave import patchNsave, patchNsaveFast_parallel

### Creating patches for SSL ###
#originalFullSizeImages_path = "D:\PHD\PhdData\FullSizeSamples/"  # This path should point to a folder with all the individual tiles/sub-images generated from the microscope (hence pointing to a "_patches" folder is intended)
originalFullSizeImages_path = "C:/Users/SH37YE/Desktop/FullSizeSamples"  # This path should point to a folder with all the individual tiles/sub-images generated from the microscope (hence pointing to a "_patches" folder is intended)
savePatches_path = "D:/PHD/PhdData/SSL_DATA_TRAINRDY_PATCHES/"  # Whereever we want the copious amount of small patches that constitute the sub-images


if __name__ == "__main__":
    images, filenames, paths = load_images_from_folders_fast(originalFullSizeImages_path, max_workers=48)
    patchNsaveFast_parallel(images, 512, 512, 0, savePatches_path, filenames, max_workers=34)
    stop = 2

for samples in os.listdir(originalFullSizeImages_path):
    #originalFullSizeImages = load_images_from_folder(originalFullSizeImages_path + "/" + samples + "/" + samples + "_patches")  # Just Loading the full size images
    #patchNsave(originalFullSizeImages, 512, 512, 0, savePatches_path, samples)  # Converts the full size images into a desired size and saves at "savePathces_path".
    stop=2