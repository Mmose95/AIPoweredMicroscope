""" The purpose of this code is to take large original images and convert them to patches manageable for processing in the following NN (and saving them to a specified folder
 This code is independent of the main and should not be run each time a new training is done. Instead, use the DataLoader to load patches directly. Note: "Dataloader" is implemented directly into each individual training loop"""
import numpy as np

#### Function for patch making ####
from itertools import product
import os


def patchNsave(images, d, e, overlapPercent, savePath, savePNGPatchs):

    patches_nd = []
    if overlapPercent > 0:
        overlapPix = int(overlapPercent/100 * d)
    else :
        overlapPix = d

    #Making patchs
    for image in images:
        patches = []
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
            #filename_all = list()
            for y in range(len(patches)): #for y in range(len(patches) // len(images)):   # Antallet af splits pr billede
                name = "{0}{1}{2}{3}{4}{5}.png".format("image", os.path.splitext(os.path.basename(image.filename))[0], "_patch_", grid[y][0], "_", grid[y][1])
                # filename = 'image ' + '%06d' + ' split' + ' %06d' % x, y  # image1 split1
                filename.append(name)
            i = 0
            for patch in patches:
                patch.save(savePath + filename[i])
                i = i + 1
                print("i:" + str(i))
    return np.array(patches_nd)

