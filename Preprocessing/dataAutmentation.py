import cv2
import numpy as np


def dataAugmentation(data):

    aug_data = []

    ##### FLIPPING ##### (Here we get x3 more data)
    for i in range(len(data)):
        aug_data.append(data(i)) #Append original patch

        # Data augmentation - flipping horizontally and vertically
        img_flipped_horiz = cv2.flip(data, 0)
        aug_data.append(img_flipped_horiz)
        img_flipped_vert = cv2.flip(data, 1)
        aug_data.append(img_flipped_vert)
        img_flipped_vert_horiz = cv2.flip(img_flipped_vert, 0)
        aug_data.append(img_flipped_vert_horiz)

    aug_data_np = np.array(aug_data)
    return aug_data_np

