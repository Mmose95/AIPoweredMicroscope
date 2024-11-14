""" This file loads the original images, patch them, and create (relatively) balanced datasets (train, validation, test) and performs data augmentation on them. """
from numpy import savez_compressed

from DataHandling.DataLoader import loadImages
from DataHandling.PatchAndSave import patchNsave

### Loading Data ###
orgImages = loadImages("E:/PhdData/Original Data/Hvidovre/10x10")

patches = patchNsave(orgImages, 256,256, 50, "E:/PhdData/InputPatches_Hvidovre/", savePNGPatches=True)

filename_input = 'test1'
savez_compressed(filename_input, patches)





