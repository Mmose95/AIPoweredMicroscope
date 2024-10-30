import os
from PIL import Image

def loadImages(path):
    images = []
    for filename in os.listdir(path):
        try:
            image = Image.open(os.path.join(path, filename))
            if image is not None:
                images.append(image)
                print('Loaded image ' + filename)
        except:
            print('FAIL: Can not load ' + filename)
    return images