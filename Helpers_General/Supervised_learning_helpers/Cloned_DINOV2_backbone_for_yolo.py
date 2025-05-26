import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from dinov2.models.vision_transformer import vit_small


class DinoV2New(nn.Module):
    def __init__(self, out_channels=384, size="small", checkpoint_path=None):
        super().__init__()

        self.out_channels = out_channels
        self.size = size

        # DINOv2 backbone
        self.backbone = vit_small(patch_size=16)
        if checkpoint_path:
            print(f"Loading weights from {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            filtered_state = {k: v for k, v in state_dict.items() if "head" not in k}
            self.backbone.load_state_dict(filtered_state, strict=False)
        else:
            print("Warning: No checkpoint path provided!")

        # Projection heads for YOLO with adaptive pooling
        self.reduce_conv1 = nn.Conv2d(out_channels, 256, kernel_size=1)
        self.reduce_conv2 = nn.Conv2d(out_channels, 512, kernel_size=1)
        self.reduce_conv3 = nn.Conv2d(out_channels, 1024, kernel_size=1)

        self.adapt1 = nn.AdaptiveAvgPool2d((16, 16))
        self.adapt2 = nn.AdaptiveAvgPool2d((8, 8))
        self.adapt3 = nn.AdaptiveAvgPool2d((4, 4))

        # Normalization
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def transform(self, x):
        b, c, h, w = x.shape
        h_new, w_new = h - h % 16, w - w % 16
        dh, dw = h - h_new, w - w_new
        x = x[:, :, dh // 2 : h - (dh - dh // 2), dw // 2 : w - (dw - dw // 2)]
        return self.normalize(x)

    def forward(self, x):
        # Skip if x already is a tuple of feature maps
        if isinstance(x, tuple):
            return x

        x = self.transform(x)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.backbone = self.backbone.to(device, dtype=dtype)
        self.reduce_conv1 = self.reduce_conv1.to(device, dtype=dtype)
        self.reduce_conv2 = self.reduce_conv2.to(device, dtype=dtype)
        self.reduce_conv3 = self.reduce_conv3.to(device, dtype=dtype)

        x = x.to(device, dtype=dtype)

        with torch.no_grad():
            tokens = self.backbone.forward_features(x)["x_norm_patchtokens"]
            b, n, d = tokens.shape  # (B, N_tokens, dim)
            hw = int(n**0.5)
            assert hw * hw == n, f"Token count {n} does not form a square"

            tokens = tokens.permute(0, 2, 1).reshape(b, d, hw, hw)

        # Adaptive pooling ensures fixed output sizes
        out1 = self.reduce_conv1(self.adapt1(tokens))  # -> [B, 256, 16, 16]
        out2 = self.reduce_conv2(self.adapt2(tokens))  # -> [B, 512, 8, 8]
        out3 = self.reduce_conv3(self.adapt3(tokens))  # -> [B, 1024, 4, 4]

        out = (out1.float(), out2.float(), out3.float())

        print("Return type:", type(out))
        for i, o in enumerate(out):
            print(f"Output {i}: Type={type(o)}, Shape={o.shape}, Dtype={o.dtype}, Device={o.device}")

        return out
