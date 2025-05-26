import torch
from torch import nn
import torch.nn.functional as F

# Dummy version of the DinoV2New class with compatible shapes
class DummyDinoV2New(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels = 384
        self.reduce_conv1 = nn.Conv2d(384, 256, kernel_size=1)
        self.reduce_conv2 = nn.Conv2d(384, 512, kernel_size=1)
        self.reduce_conv3 = nn.Conv2d(384, 1024, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        # Simulate transformer patch output: 14x14 patches => 196 tokens
        x = torch.randn(B, 384, 16, 16)
        out1 = self.reduce_conv1(x)
        out2 = self.reduce_conv2(F.avg_pool2d(x, 2))  # 8x8
        out3 = self.reduce_conv3(F.avg_pool2d(x, 4))  # 4x4
        return [out1, out2, out3]

# Instantiate and test with dummy input
model = DummyDinoV2New()
dummy_input = torch.randn(1, 3, 224, 224)  # standard image size
feature_maps = model(dummy_input)

import pandas as pd

# Create a DataFrame of feature map shapes
shapes = [tuple(fm.shape) for fm in feature_maps]
df = pd.DataFrame(shapes, columns=["Batch", "Channels", "Height", "Width"])
print(df)

