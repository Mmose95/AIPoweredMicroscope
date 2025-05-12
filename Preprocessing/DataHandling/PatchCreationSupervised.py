""" This file loads the original images, patch them, and create (relatively) balanced datasets (train, validation, test) and performs data augmentation on them. """
import json
import os

from Helpers_General.LoadImages import load_images_from_folders_fast
from Helpers_General.PatchAndSave import patchNsave
from Preprocessing.DataHandling.VizCocoBbox import display_images_with_coco_annotations
from configSetting import load_settings

def PatchCreationSupervised(CreateOrLoadPatches, vizLabels = False):

    originalFullSizeImages_path, savePatches_path, originalPatchedImages_path, _, cocoFormat_patched_labels_path = load_settings()

    if CreateOrLoadPatches in ['Create']:
        ### Loading Data ###
        originalFullSizeImages = load_images_from_folders_fast(originalFullSizeImages_path)
        patches = patchNsave(originalFullSizeImages, 256, 256, 50, savePatches_path, savePNGPatchs=True)  # UNI = "D:/PhdData/Bbox/Patches/"

    if CreateOrLoadPatches in ['Load']:
        """ Load patches if already made (and convert to ndarray) """
        # patches = loadImages(originalPatchedImages_path)
        # patch_arrays = [np.array(patch) for patch in patches]
        # patches = np.stack(patch_arrays, axis=0)

    """ If we want to save input data for another time in an appropriate format (.npz (or other)) """
    #filename_input = 'test1'
    #savez_compressed(filename_input, patches)

    # Load COCO annotations
    with open(cocoFormat_patched_labels_path, 'r') as f:annotations = json.load(f)
    # Get all image files
    all_image_files = [os.path.join(originalPatchedImages_path, img['file_name'].replace('.tif', '')) for img in annotations['images']]
    # random_image_files = random.sample(all_image_files, 5) #Set number to vizualise
    random_image_files = all_image_files[0:4]


    # Choose between 'bbox', 'seg', or 'both'
    display_type = 'bbox'
    if vizLabels == True:
        display_images_with_coco_annotations(random_image_files, annotations, display_type)

    ''''Here we could create a sorting algorithm that splits data into train, test and val'''

    dataset = [all_image_files,annotations]

    return dataset