
from torch.utils.data import Dataset
from PIL import Image
import os
import glob

class UnlabeledImageFolder(Dataset):
    def __init__(self, root, transform=None, **kwargs):
        self.image_paths = sorted(glob.glob(os.path.join(root, "**", "*.*"), recursive=True))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            crops = self.transform(img)  # should return a dict
        else:
            crops = {"global_crops": [img], "local_crops": []}
        return crops