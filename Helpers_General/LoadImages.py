import os
from PIL import Image

'''def load_images_from_folder(path):
    images = []
    i = 0
    valid_extensions = ('.png', '.tif', '.tiff')
    image_files = [f for f in os.listdir(path) if f.lower().endswith(valid_extensions)]
    file_count = len(image_files)

    for filename in image_files:
        file_path = os.path.join(path, filename)
        i += 1
        with Image.open(file_path) as img:
            loaded_img = img.copy()
            loaded_img.filename = filename
            images.append(loaded_img)
        print(f'Loaded image {i}/{file_count}: {filename}')

    return images'''

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
import imageio.v3 as iio

def _load_image_as_array(file_path):
    try:
        img = iio.imread(file_path)
        return img, file_path.name, file_path
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return None

def load_images_from_folders_fast(root_folder, extensions=('.tif', '.tiff', '.png'), max_workers=24):
    root = Path(root_folder)
    patch_dirs = []

    # Look inside each numbered folder and check for a matching *_patches folder
    for sample_folder in root.iterdir():
        if sample_folder.is_dir():
            patch_subfolder = sample_folder / f"{sample_folder.name}_patches"
            if patch_subfolder.exists():
                patch_dirs.append(patch_subfolder)

    print(f"🔍 Found {len(patch_dirs)} '_patches' subfolders in: {root}")

    image_paths = []
    for folder in patch_dirs:
        for ext in extensions:
            image_paths.extend(folder.rglob(f"*{ext}"))

    print(f"📂 Found {len(image_paths)} images. Loading...")

    images = []
    filenames = []
    paths = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_load_image_as_array, image_paths), total=len(image_paths)))


    for result in results:
        if result:
            img, name, path = result
            images.append(img)
            filenames.append(name)
            paths.append(path)

    print(f"✅ Loaded {len(images)} total images from all patch subfolders.")
    return images, filenames, paths
