from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torch

def setup_coco_loader(images_dir, annotations_json, batch_size, device):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    dataset = CocoDetection(images_dir, annotations_json, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def compute_detection_loss(preds, targets):
    classification_preds, bbox_preds = preds
    classification_targets, bbox_targets = targets

    cls_loss = F.cross_entropy(classification_preds, classification_targets)
    bbox_loss = F.mse_loss(bbox_preds, bbox_targets)

    return cls_loss + bbox_loss


def evaluate_map50(encoder, detection_head, data_loader, device):
    encoder.eval()
    detection_head.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            features = encoder(images)
            preds = detection_head(features)

            all_preds.extend(preds)
            all_targets.extend(targets)

    map50 = calculate_map50(all_preds, all_targets)  # Implement or reuse existing metric calculation
    return map50

import torch
from pathlib import Path
import mlflow
import pandas as pd
from datetime import datetime
from rfdetr.detection_head import RFDETRHead  # replace with your actual import path
from dinov2.models.vision_transformer import vit_base
from Utils_MLFLOW import setup_mlflow_experiment

def run_linear_probe_all(ssl_data_path, save_folder, current_SSL_RunID):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    img_dir = Path(ssl_data_path) / "images"
    annotations = Path(ssl_data_path) / "_annotations.coco.json"
    checkpoints = sorted(Path(save_folder).glob("*.pt"))

    # MLflow tracking
    experiment_id = setup_mlflow_experiment("Linear Probing for SSL Encoder Selection")
    run_name = f"LinearProbing_{current_SSL_RunID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    results = []

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        for ckpt in checkpoints:
            model_name = ckpt.stem
            model_type = 'teacher' if 'teacher' in model_name.lower() else 'student'

            mlflow.log_param(f"Checkpoint_{model_name}", str(ckpt))

            # Load SSL pretrained ViT encoder
            encoder = vit_base(patch_size=16, img_size=224)
            state_dict = torch.load(ckpt, map_location=device)['model']
            encoder.load_state_dict(state_dict, strict=False)
            encoder.eval().requires_grad_(False).to(device)

            # Load detection head
            detection_head = RFDETRHead(num_classes=4).to(device)  # Adjust num_classes as needed
            optimizer = torch.optim.Adam(detection_head.parameters(), lr=1e-3)

            # Data Loader (COCO format)
            train_loader = setup_coco_loader(img_dir, annotations, batch_size=32, device=device)

            # Probe training (small epochs for quick evaluation)
            epochs = 10
            for epoch in range(epochs):
                detection_head.train()
                for images, targets in train_loader:
                    optimizer.zero_grad()
                    features = encoder(images)
                    preds = detection_head(features)
                    loss = compute_detection_loss(preds, targets)  # implement this helper
                    loss.backward()
                    optimizer.step()

            # Evaluate mAP@50
            val_loader = setup_coco_loader(img_dir, annotations, batch_size=32, device=device)
            map50 = evaluate_map50(encoder, detection_head, val_loader, device)
            mlflow.log_metric(f"mAP50_{model_name}", map50)

            results.append({"checkpoint": str(ckpt), "mAP50": map50, "model_type": model_type})

        # Save results and select best
        df_results = pd.DataFrame(results)
        results_csv = Path(save_folder) / "linear_probing_results.csv"
        df_results.to_csv(results_csv, index=False)
        mlflow.log_artifact(str(results_csv))

        best_checkpoint = df_results.loc[df_results["mAP50"].idxmax(), "checkpoint"]
        best_path_file = Path(save_folder) / "best_encoder_path.txt"
        best_path_file.write_text(best_checkpoint)
        mlflow.log_artifact(str(best_path_file))

        print(f"✅ Best SSL encoder selected based on mAP@50: {best_checkpoint}")

    return best_checkpoint
