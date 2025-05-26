# dinov2_backbone.py
import torch
import torch.nn as nn
from dinov2.models.vision_transformer import vit_small  # adapt path to your model

class DINOv2Backbone_DINO(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = vit_small(patch_size=16)

        # Allow path from args or hardcode for now
        state_dict = torch.load(args.pretrained_path, map_location="cpu")
        state_dict = {k: v for k, v in state_dict.items() if "head" not in k}
        self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.backbone.forward_features(x)  # Returns patch tokens (B, N, C)


