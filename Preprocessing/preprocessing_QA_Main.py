""" This file loads the original images, patch them, and create (relatively) balanced datasets (train, validation, test) and performs data augmentation on them. """
import json
import os
import random

from numpy import savez_compressed
from DataHandling.DataLoader import loadImages
from DataHandling.PatchAndSave import patchNsave
from DataHandling.VizCocoBbox import display_images_with_coco_annotations

### Loading Data ###
#orgImages = loadImages("X:/PhdData/BBox/Original/")

#patches = patchNsave(orgImages, 256,256, 50, "X:/PhdData/BBox/Patches/", savePNGPatchs=True)

#filename_input = 'test1'
#savez_compressed(filename_input, patches)

# Load COCO annotations
with open("C:/Users/mose_/AAU/Sundhedstek/PhD/AIPoweredMicroscope_development/DataHandling/coco_patched_annotations.json", 'r') as f:
    annotations = json.load(f)

# Get all image files
image_dir = "X:/PhdData/BBox/Patches/"
all_image_files = [os.path.join(image_dir, img['file_name']) for img in annotations['images']]
random_image_files = random.sample(all_image_files, 5)

# Choose between 'bbox', 'seg', or 'both'
display_type = 'bbox'
display_images_with_coco_annotations(random_image_files, annotations, display_type)







