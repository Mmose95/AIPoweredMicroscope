from Helpers_General.LoadImages import loadImages
from Helpers_General.PatchAndSave import patchNsave
from MainPhase_QualityAssessment.MainPhase_QualityAssessment import qualityAssessment

##Switches/Modes for training different networks.
train_QualityAssessment = True
train_SpeciesDetermination = False


"""Initialize MLflow server for capturing experiments (remember to set "track experiment = true)"""

#subprocess.Popen(["StartMLflowServer.cmd"], shell=True)

########################## Phase 1: Quality Assessment ##########################
if train_QualityAssessment == True:

    '''Preparation functions for QualityAssessment'''

    ### Creating patches for SSL ###
    originalFullSizeImages_path = "D:\PHD\PhdData\FullSizeSamples/113331239355/113331239355_patches" #This path should point to a folder with all the individual tiles/sub-images generated from the microscope (hence pointing to a "_patches" folder is intended)
    savePatches_path = "D:/PHD/PhdData/SSL_DATA_PATCHES/" #Whereever we want the copious amount of small patches that constitute the sub-images

    originalFullSizeImages = loadImages(originalFullSizeImages_path)
    patches = patchNsave(originalFullSizeImages, 512, 512, 0, savePatches_path, savePNGPatchs=True)

    ### Convert labeled patches to match input patches ###


    '''training process'''
    qualityAssessment(trackExperiment_SSL=True)

stop = 1

########################## Phase 2: Species Determination ##########################

if train_SpeciesDetermination == True:
 bob = 1
