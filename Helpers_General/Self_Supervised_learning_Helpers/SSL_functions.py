""" Function of the script:
More or less randomly holds scripts that is used in relation to SSL learning/training.
Does not directly include versions of architectures and such.
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import random

class MicroscopyDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

# Define Transformations for SSL
ssl_transform = transforms.Compose([
    transforms.RandomResizedCrop(518), #this will become the final size of the patches (518 fits the DINOv2 pretrained model)

    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Define a simple PatchEmbedding module ---
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

# --- Dummy Dataset Example ---
class DummyImageDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=224):
        self.num_samples = num_samples
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a dummy image using PIL (here, a random noise image)
        img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
        return self.transform(img)


def patchify(imgs, patch_size=16):
    """
    Splits an image into patches.

    Args:
        imgs: Tensor of shape (B, C, H, W)
        patch_size: The size of each patch (assumed square)

    Returns:
        patches: Tensor of shape (B, num_patches, patch_size*patch_size*C)
    """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0, \
        "Image dimensions must be divisible by the patch size."
    h_patches = H // patch_size
    w_patches = W // patch_size

    # First, reshape to separate the patches:
    # From (B, C, H, W) to (B, C, h_patches, patch_size, w_patches, patch_size)
    patches = imgs.reshape(B, C, h_patches, patch_size, w_patches, patch_size)

    # Permute to bring patch dimensions together:
    # From (B, C, h_patches, patch_size, w_patches, patch_size) to
    # (B, h_patches, w_patches, C, patch_size, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5)

    # Finally, reshape to combine patch grid and patch content:
    # From (B, h_patches, w_patches, C, patch_size, patch_size) to
    # (B, h_patches*w_patches, C*patch_size*patch_size)
    patches = patches.reshape(B, h_patches * w_patches, C * patch_size * patch_size)

    return patches



# Example unpatchify if needed:
def unpatchify(patches, patch_size=16, img_size=224):
    """
    patches: (B, num_patches, patch_dim) with patch_dim = C*patch_size*patch_size
    Returns:
      imgs: (B, C, H, W)
    """
    B, num_patches, patch_dim = patches.shape
    C = patch_dim // (patch_size * patch_size)
    h_patches = w_patches = int(num_patches**0.5)
    imgs = patches.reshape(B, h_patches, w_patches, C, patch_size, patch_size)
    imgs = imgs.permute(0, 3, 1, 4, 2, 5)
    imgs = imgs.reshape(B, C, h_patches * patch_size, w_patches * patch_size)
    return imgs

from torch.utils.data import Dataset
from torchvision import transforms

class PILImageDataset(Dataset):
    def __init__(self, images, transform=None):
        """
        Args:
            images (list): A list of PIL images.
            transform (callable, optional): A function/transform to apply to each image.
        """
        self.images = images
        if transform is None:
            # Default transform: resize and convert to tensor.
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Adjust size as needed.
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image