
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel

# ---------------------------
# 1) choose model
# ---------------------------
# Best simple starting point:
#  = "facebook/dinov2-with-registers-small"

# If you later get access to DINOv3, you can try:
MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"

IMAGE_PATH = "D:\PHD\PhdData\CellScanData\Zoom10x - Quality Assessment_Cleaned\Sample 55\Patches for Sample 55\Sample55 - BF.0_0_patch_x0_y560.tif"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# 2) load image
# ---------------------------
image = Image.open(IMAGE_PATH).convert("RGB")

# ---------------------------
# 3) load processor + model
# ---------------------------
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ---------------------------
# 4) preprocess + forward pass
# ---------------------------
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.inference_mode():
    outputs = model(**inputs)

last_hidden = outputs.last_hidden_state  # [B, tokens, dim]
print("last_hidden_state shape:", tuple(last_hidden.shape))

# ---------------------------
# 5) separate CLS / register / patch tokens
# ---------------------------
patch_size = model.config.patch_size
num_registers = getattr(model.config, "num_register_tokens", 0)

_, _, H, W = inputs["pixel_values"].shape
nH, nW = H // patch_size, W // patch_size

cls_token = last_hidden[:, 0, :]  # [1, dim]
patch_tokens = last_hidden[:, 1 + num_registers:, :]  # [1, n_patches, dim]

print("CLS token shape:", tuple(cls_token.shape))
print("Patch tokens shape:", tuple(patch_tokens.shape))
print("Patch grid:", (nH, nW))

patch_grid = patch_tokens[0].reshape(nH, nW, -1).cpu().numpy()  # [nH, nW, dim]

# ---------------------------
# 6) visualize representation with PCA -> RGB
# ---------------------------
flat = patch_grid.reshape(-1, patch_grid.shape[-1])
pca = PCA(n_components=3)
rgb = pca.fit_transform(flat)

# normalize each component to [0,1]
rgb = (rgb - rgb.min(axis=0, keepdims=True)) / (
    rgb.max(axis=0, keepdims=True) - rgb.min(axis=0, keepdims=True) + 1e-8
)
rgb = rgb.reshape(nH, nW, 3)

# ---------------------------
# 7) cosine similarity map from one chosen patch
# ---------------------------
# choose center patch
cy, cx = nH // 2, nW // 2
anchor = patch_grid[cy, cx]  # [dim]

norm_grid = patch_grid / (np.linalg.norm(patch_grid, axis=-1, keepdims=True) + 1e-8)
norm_anchor = anchor / (np.linalg.norm(anchor) + 1e-8)

cos_map = (norm_grid * norm_anchor).sum(axis=-1)

# ---------------------------
# 8) show results
# ---------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(image)
axes[0].set_title("Original image")
axes[0].axis("off")

axes[1].imshow(inputs["pixel_values"][0].permute(1, 2, 0).cpu().numpy())
axes[1].set_title(f"Preprocessed ({H}x{W})")
axes[1].axis("off")

axes[2].imshow(rgb)
axes[2].set_title("Patch features (PCA -> RGB)")
axes[2].axis("off")

im = axes[3].imshow(cos_map, cmap="viridis")
axes[3].scatter([cx], [cy], c="red", s=40)
axes[3].set_title("Cosine similarity to center patch")
axes[3].axis("off")
plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()


'''
from huggingface_hub import hf_hub_download, whoami

print(whoami(token=token))

print(hf_hub_download(
    repo_id="facebook/dinov3-vits16-pretrain-lvd1689m",
    filename="preprocessor_config.json",
    token=token,
))

print(hf_hub_download(
    repo_id="facebook/dinov3-vits16-pretrain-lvd1689m",
    filename="config.json",
    token=token,
))
'''
'''
from huggingface_hub import whoami
from transformers import AutoImageProcessor, AutoModel

print("HF user:", whoami(token=token))

model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"

processor = AutoImageProcessor.from_pretrained(model_id, token=token)
model = AutoModel.from_pretrained(model_id, token=token)

print("loaded ok")
print(type(processor))
print(type(model))'''
