from torch.utils.data import Dataset
from PIL import Image
import os
import glob

class SSLImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        for ext in ("*.jpg", "*.png", "*.jpeg", "*.tif"):
            all_matches = glob.glob(os.path.join(root_dir, "**", ext), recursive=True)
            # ❌ Exclude any path that contains a "_rejected" folder
            filtered = [p for p in all_matches if "_rejected" not in p]
            self.image_paths.extend(filtered)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in: {root_dir}")

        print(f"[INFO] Found {len(self.image_paths)} images")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            output = self.transform(image)
            image = output["global_crops"]

        return image, 0  # image is now a list of tensors (global crops)