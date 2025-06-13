import os
import json
import shutil
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import mlflow
from tqdm import tqdm
import pandas as pd

from Utils_MLFLOW import setup_mlflow_experiment
from dinov2.models.vision_transformer import vit_base

# ---- Dataset Class ----
class PurePatchDataset(Dataset):
    def __init__(self, images_dir, json_path, transform=None):
        with open(json_path, 'r') as f:
            self.label_map = json.load(f)
        self.images_dir = Path(images_dir)
        self.samples = list(self.label_map.items())
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = self.images_dir / fname
        image = self.transform(transforms.functional.pil_to_tensor(transforms.functional.pil_to_tensor(transforms.functional.to_pil_image(torch.load(img_path)))))
        return image.float(), int(label)

# ---- Linear Probe Model ----
class LinearProbe(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.encoder(x)
        return self.head(x)

# ---- Probing Function ----
def evaluate_model(checkpoint_path, dataset, device, num_classes=4, batch_size=32, epochs=20):
    results = {}
    tag = Path(checkpoint_path).stem
    model_type = 'teacher' if 'teacher' in tag.lower() else 'student'

    encoder = vit_base(patch_size=16, img_size=224)
    state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()
    encoder.requires_grad_(False)

    model = LinearProbe(encoder, num_classes).to(device)
    optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    with mlflow.start_run(run_name=f"linear_probe_{tag}", nested=True):
        mlflow.log_param("checkpoint", checkpoint_path)
        mlflow.log_param("model_type", model_type)

        for epoch in range(epochs):
            model.train()
            total, correct, total_loss = 0, 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                total_loss += loss.item()

            acc = correct / total
            mlflow.log_metric("accuracy", acc, step=epoch)
            mlflow.log_metric("loss", total_loss / len(loader), step=epoch)

        # Final accuracy
        results['checkpoint'] = checkpoint_path
        results['model_type'] = model_type
        results['accuracy'] = acc
        return results

# ---- Run All Checkpoints ----
def run_linear_probe_all(ssl_data_path, save_folder, current_SSL_RunID):
    img_dir = Path(ssl_data_path) / "images"
    json_path = Path(save_folder) / "pure_patch_class_map_balanced.json"
    dataset = PurePatchDataset(img_dir, json_path)
    ckpts = sorted(Path(save_folder).glob("*student.pt")) + sorted(Path(save_folder).glob("*teacher.pt"))

    with mlflow.start_run(run_name=f"Linear Probing for {current_SSL_RunID}_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}", nested=True) as run:
        nested_run_folder = Path(save_folder) / run.info.run_name
        nested_run_folder.mkdir(exist_ok=True, parents=True)

        # Copy JSON for traceability
        shutil.copy(json_path, nested_run_folder)

        all_results = []
        for ckpt in ckpts:
            result = evaluate_model(str(ckpt), dataset, device="cuda")
            all_results.append(result)

        # Save results
        out_dir = nested_run_folder
        out_dir.mkdir(exist_ok=True, parents=True)
        df = pd.DataFrame(all_results)
        df.to_csv(out_dir / "results.csv", index=False)

        # Best model selection
        best = df.sort_values("accuracy", ascending=False).iloc[0]
        best_path = best['checkpoint']
        with open(out_dir / "best_encoder_path.txt", "w") as f:
            f.write(best_path)
        print(f"✅ Best model selected: {best_path}")

    return best_path
