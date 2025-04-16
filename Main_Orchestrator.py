import warnings
import logging

# Suppress warnings before any other import triggers them
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", message=".*No module named 'triton'.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
logging.getLogger().setLevel(logging.ERROR)

from Helpers_General.LoadImages import load_images_from_folder
from Helpers_General.PatchAndSave import patchNsave
from MainPhase_QualityAssessment.MainPhase_QualityAssessment import qualityAssessment_SSL
import sys

##Switches/Modes for training different networks.
train_QualityAssessment_SSL = True
CreatePatchesFromFullSizeImages = False

train_QualityAssessment_Supervised = False

train_SpeciesDetermination = False


"""Initialize MLflow server for capturing experiments (remember to set "track experiment = true)"""

#subprocess.Popen(["StartMLflowServer.cmd"], shell=True)

########################## Phase 1: Quality Assessment ##########################

sys.path.append("dinov2")

if train_QualityAssessment_SSL == True:

    '''Preparation functions for QualityAssessment'''

    ### Creating patches for SSL ###
    originalFullSizeImages_path = "D:\PHD\PhdData\FullSizeSamples/113331239355/113331239355_patches" #This path should point to a folder with all the individual tiles/sub-images generated from the microscope (hence pointing to a "_patches" folder is intended)
    savePatches_path = "D:/PHD/PhdData/SSL_DATA_PATCHES/" #Whereever we want the copious amount of small patches that constitute the sub-images

    if CreatePatchesFromFullSizeImages == True:
        originalFullSizeImages = load_images_from_folder(originalFullSizeImages_path) #Just Loading the full size images
        patches = patchNsave(originalFullSizeImages, 512, 512, 0, savePatches_path, savePNGPatchs=True) #Converts the full size images into a desired size and saves at "savePathces_path".

    '''training process'''
    if __name__ == "__main__":
        import multiprocessing
        multiprocessing.freeze_support()  # Optional, safe to add
        qualityAssessment_SSL(False, "D:/PHD/PhdData/SSL_DATA_PATCHESTest")

if train_QualityAssessment_Supervised == True:

    something = 1

stop = 1

########################## Phase 2: Species Determination ##########################

if train_SpeciesDetermination == True:
 bob = 1
