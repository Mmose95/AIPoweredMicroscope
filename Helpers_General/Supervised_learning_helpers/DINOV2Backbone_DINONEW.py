import torch.nn as nn
import torch
from DINO.util.misc import NestedTensor

class DinoV2ForDINONEW(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Conv2d(384, 256, kernel_size=1)  # vit_small has 384 dim
        self.num_channels = [256] * 4  # DINO expects a list of output dims for each feature level

    def forward(self, x):
        if isinstance(x, NestedTensor):
            x = x.tensors  # unwrap from NestedTensor

        B, C, H, W = x.shape
        out = self.backbone(x)

        # if backbone returns a tuple/list, take last
        if isinstance(out, (list, tuple)):
            feat = out[-1]
        else:
            feat = out

        # Handle different output formats
        if feat.dim() == 2:
            # Calculate number of patches based on feature size
            feat_size = feat.size(-1)  # Get the feature dimension
            N = feat.size(0) // B  # Calculate N from batch and total size
            feat = feat.view(B, N, -1)
        elif feat.dim() == 4:
            # If output is already in spatial format [B, C, H, W]
            B, C, H, W = feat.shape
            feat = feat.flatten(2).transpose(1, 2)  # [B, N, C]

        # Drop CLS token if present
        if feat.size(1) > (H//16 * W//16):
            feat = feat[:, 1:, :]

        # Calculate proper spatial dimensions
        N = feat.size(1)
        hw = int(N ** 0.5)
        
        # Reshape maintaining the correct dimensions
        feat = feat.transpose(1, 2).contiguous().view(B, -1, hw, hw)

        # Return multiple feature levels
        projected_feat = self.input_proj(feat)
        mask = torch.zeros((B, hw, hw), dtype=torch.bool, device=feat.device)  # no padding

        nested_feats = []
        for i in range(4):
            scaled = nn.functional.interpolate(projected_feat, scale_factor=1 / (2 ** i), mode='bilinear',
                                               align_corners=False)
            mask_scaled = nn.functional.interpolate(mask[None].float(), size=scaled.shape[-2:]).to(torch.bool)[0]
            nested_feats.append(NestedTensor(scaled, mask_scaled))

        return nested_feats, None