""" Function of the script:
Holds scripts/code-snippets/functions directly related to SSL architecture construction/setup.
"""

import torch
import torch.nn as nn

from Helpers_General.Self_Supervised_learning_Helpers.SSL_functions import PatchEmbedding

'''Class NameOfFunc(inheritance):
    def __init-- (self, inputs) 
        super().__init__()
        this here defines the layers: 
        layer1 = .... 
        self.name = nn.layer(layer1, num_layers = ... ) 
'''

# --- simple MAE Encoder and Decoder --- (used in first tests)
class MAEEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=12):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=768, decoder_dim=512, depth=4, num_patches=196):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_proj = nn.Linear(embed_dim, decoder_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_dim, nhead=8)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)
        self.output_proj = nn.Linear(decoder_dim, 16*16*3)  # Assuming patch size 16, RGB

    def forward(self, x, ids_restore):
        # x: [B, num_visible, embed_dim]
        B, L, _ = x.shape
        x = self.decoder_proj(x)

        # Create mask tokens for missing patches
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - L, 1)
        # Reconstruct full sequence: place tokens at positions defined by ids_restore
        x_full = torch.zeros(B, ids_restore.shape[1], x.shape[-1], device=x.device)
        x_full.scatter_(1, ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[-1]), torch.cat([x, mask_tokens], dim=1))
        # Decode
        x_decoded = self.decoder(tgt=x_full, memory=x_full)
        # Predict pixel values for each patch
        x_reconstructed = self.output_proj(x_decoded)
        return x_reconstructed


# --- Simple MAE model combining encoder and decoder ---
class MAEModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 encoder_depth=12, decoder_dim=512, decoder_depth=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.encoder = MAEEncoder(embed_dim, encoder_depth)
        num_patches = (img_size // patch_size) ** 2
        self.decoder = MAEDecoder(embed_dim, decoder_dim, decoder_depth, num_patches)

    def random_masking(self, x, mask_ratio=0.75):
        B, L, _ = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        # Create a placeholder for restoration indices
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        return x_masked, ids_restore

    def forward(self, x, mask_ratio=0.75):
        x = self.patch_embed(x)
        x_masked, ids_restore = self.random_masking(x, mask_ratio)
        latent = self.encoder(x_masked)
        x_reconstructed = self.decoder(latent, ids_restore)
        return x_reconstructed