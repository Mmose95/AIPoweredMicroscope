import os
from PIL import Image

def load_images_from_folder(path):
    images = []
    i = 0
    png_files = [f for f in os.listdir(path) if f.lower().endswith('.png')]
    file_count = len(png_files)

    for filename in png_files:
        file_path = os.path.join(path, filename)
        i += 1
        with Image.open(file_path) as img:
            loaded_img = img.copy()
            images.append(loaded_img)
        print('Loaded image', i, "/", file_count)

    return images