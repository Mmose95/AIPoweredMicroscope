import torch
import torch.nn as nn
from dinov2.models.vision_transformer import vit_small

class DINOv2Backbone_YOLO(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()

        print("CUDA Available:", torch.cuda.is_available())
        print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

        self.backbone = vit_small(patch_size=16)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        filtered_state = {k: v for k, v in state_dict.items() if "head" not in k}
        self.backbone.load_state_dict(filtered_state, strict=False)

        self.fpn1 = nn.Conv2d(384, 256, kernel_size=1)
        self.fpn2 = nn.Conv2d(384, 512, kernel_size=3, stride=2, padding=1)
        self.fpn3 = nn.Conv2d(384, 1024, kernel_size=3, stride=4, padding=1)

        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

        if torch.cuda.is_available():
            self.cuda()
            print("DINOv2Backbone moved to: cuda")
        else:
            print("DINOv2Backbone staying on: cpu")

        self.out_channels = (256, 512, 1024)

    def forward(self, x):
        x = x.to(self.fpn1.weight.device)
        out = self.backbone.forward_features(x)
        x = out["x_norm_patchtokens"]
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        # Apply FPNs
        fpn1_out = self.upsample2(self.fpn1(x))  # becomes 48x48 (stride 8)
        fpn2_out = self.upsample1(self.fpn2(x))  # becomes 24x24 (stride 16)
        fpn3_out = self.fpn3(x)

        print(f"[DINOv2Backbone] Output shapes: {fpn1_out.shape}, {fpn2_out.shape}, {fpn3_out.shape}")
        return fpn1_out, fpn2_out, fpn3_out