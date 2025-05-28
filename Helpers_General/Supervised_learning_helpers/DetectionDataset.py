import os
from torch.utils.data import Dataset
from PIL import Image
import torch

from DINO.datasets import transforms as dino_transforms


class DetectionDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = sorted([
            f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir,
                                  self.image_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))

        image = Image.open(img_path).convert("RGB")
        w, h = image.size  # note: PIL image size is (W, H)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, cx, cy, bw, bh = map(float, parts)
                        x_min = (cx - bw / 2) * w
                        y_min = (cy - bh / 2) * h
                        x_max = (cx + bw / 2) * w
                        y_max = (cy + bh / 2) * h
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(int(class_id))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "orig_size": torch.tensor([h, w]),
            "size": torch.tensor([h, w]),
        }

        # ✅ Apply both image and target transform
        image, target = self.transform(image, target)

        return image, target

