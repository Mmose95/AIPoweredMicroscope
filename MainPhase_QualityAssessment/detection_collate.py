import torch
import torch.nn.functional as F
from util import NestedTensor

class DetectionCollateFn:
    def __init__(self, model):
        # Accepts the DINOv2 core encoder
        self.model = model

    def __call__(self, batch):
        images, targets = zip(*batch)
        #images = torch.stack(images, dim=0)
        nested_images = self.prepare_dino_input(images)
        return nested_images, list(targets)

    def prepare_dino_input(self, batch):
        _, _, h, w = batch.shape

        try:
            patch_size = getattr(self.model.encoder.config, "patch_size", 14)
            num_windows = getattr(self.model.encoder.config, "num_windows", 4)
        except AttributeError:
            patch_size = 14
            num_windows = 4

        multiple = patch_size * num_windows
        new_h = ((h + multiple - 1) // multiple) * multiple
        new_w = ((w + multiple - 1) // multiple) * multiple

        padding = [0, new_w - w, 0, new_h - h]
        padded_images = F.pad(batch, padding, value=0)

        mask = torch.zeros((batch.shape[0], h, w), dtype=torch.bool, device=batch.device)
        padded_mask = F.pad(mask, padding, value=True)

        return NestedTensor(padded_images, padded_mask)