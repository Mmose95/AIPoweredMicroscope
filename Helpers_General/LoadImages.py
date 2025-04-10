import os
from PIL import Image

def load_images_from_folder(path):
    images = []
    i = 0
    totFiles = os.listdir(path)
    file_count = len(totFiles)
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        i = i + 1
        with Image.open(file_path) as img:
            loaded_img = img.copy()
            images.append(loaded_img)
        print('Loaded image ' , i , "/" , file_count)

    return images