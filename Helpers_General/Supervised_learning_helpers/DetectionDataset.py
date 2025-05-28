import torch
from torch.utils.data import Dataset
from PIL import Image
import os

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
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, os.path.splitext(self.image_files[idx])[0] + ".txt")

        image = Image.open(image_path).convert("RGB")
        target = self._load_yolo_annotation(label_path, image.size)

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def _load_yolo_annotation(self, label_path, image_size):
        width, height = image_size
        boxes = []
        labels = []

        if not os.path.exists(label_path):
            return {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64)}

        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                class_id, x_center, y_center, w, h = map(float, parts)

                # Convert normalized YOLO format to absolute corner format
                x_center *= width
                y_center *= height
                w *= width
                h *= height

                x_min = x_center - w / 2
                y_min = y_center - h / 2
                x_max = x_center + w / 2
                y_max = y_center + h / 2

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(class_id))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return target
