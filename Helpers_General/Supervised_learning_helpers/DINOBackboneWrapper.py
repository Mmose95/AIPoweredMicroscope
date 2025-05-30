import torch
import torch.nn as nn
from DINO.util.misc import NestedTensor

class DINOBackboneWrapper(nn.Module):
    def __init__(self, backbone, position_embedding):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.num_channels = backbone.num_channels  # Expose this for build_dino

    def forward(self, x):
        # DINO expects a tuple: (features, positional_embeddings)
        # Convert x into NestedTensor if needed
        if not isinstance(x, NestedTensor):
            x = NestedTensor(x, torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.bool, device=x.device))

        features, _ = self.backbone(x.tensors)  # feature maps as list of tensors


        pos = [self.position_embedding(nf).to(dtype=nf.tensors.dtype) for nf in features]

        return features, pos




    def __getitem__(self, idx):
        if idx == 0:
            return self.backbone
        elif idx == 1:
            return self.position_embedding
        else:
            raise IndexError("DINOBackboneWrapper only supports indices 0 (backbone) and 1 (positional encoding).")
