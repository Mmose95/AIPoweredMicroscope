""" The purpose of this code is to take large original images and convert them to patches manageable for processing in the following NN (and saving them to a specified folder
 This code is independent of the main and should not be run each time a new training is done. Instead, use the DataLoader to load patches directly"""
import numpy as np
from numpy import savez_compressed

from DataHandling.DataLoader import loadImages
from numpy import savez_compressed
from numpy import asarray
#### Function for patch making ####
from itertools import product
import os


def patchNsave(images, d, e, overlapPercent, savePath, savePNGPatchs):

    patches = []
    patches_nd = []
    overlapPix = int(overlapPercent/100 * d)

    #Making patchs
    for image in images:
        w, h = image.size
        grid = list(product(range(0, h - h % d, overlapPix), range(0, w - w % e, overlapPix)))
        for i, j in grid:
            box = (j, i, j + d, i + e)
            patch = image.crop(box)
            patches.append(patch)
            patches_nd.append(np.array(patch))


    #Saving to folder
    if savePNGPatchs == True:
        filename = list()
        for x in range(len(images)):
            for y in range(len(patches) // len(images)):  # Antallet af splits pr billede
                name = "{0}{1}{2}{3}.png".format("image", os.path.splitext(os.path.basename(images[x].filename))[0], "split", y + 1)
                # filename = 'image ' + '%06d' + ' split' + ' %06d' % x, y  # image1 split1
                filename.append(name)
        i = 0
        for patch in patches:
            patch.save(savePath + filename[i])
            i = i + 1
            print("i:" + str(i))
        return np.array(patches_nd)

