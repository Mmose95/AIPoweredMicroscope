import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import mlflow
import pandas as pd
from pathlib import Path
from dinov2.models.vision_transformer import vit_base
from Utils_MLFLOW import setup_mlflow_experiment
from datetime import datetime
import torch.nn as nn

# Lightweight detection head
class SimpleRFDETRHead(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.bbox_regressor = nn.Linear(embed_dim, 4)

    def forward(self, features):
        class_logits = self.classifier(features)
        bbox_preds = self.bbox_regressor(features)
        return class_logits, bbox_preds

def xywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(-1)
    x1, y1 = x_c - w / 2, y_c - h / 2
    x2, y2 = x_c + w / 2, y_c + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)

def extract_boxes_labels(target):
    boxes = target["bbox"]  # [[x,y,w,h], ...]
    labels = target["category_id"]  # [label1, label2, ...]

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return boxes_tensor, labels_tensor


def setup_coco_loader(images_dir, annotations_json, batch_size):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    dataset = CocoDetection(images_dir, annotations_json, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

def compute_detection_loss(preds, targets):
    class_preds, bbox_preds = preds

    class_targets = []
    bbox_targets = []

    for target in targets:
        labels = target["category_id"]
        boxes = target["bbox"]

        for label, box in zip(labels, boxes):
            class_targets.append(label)
            bbox_targets.append(torch.tensor(box))

    class_targets = torch.tensor(class_targets, dtype=torch.long, device=class_preds.device)
    bbox_targets = torch.stack(bbox_targets).to(dtype=torch.float32, device=bbox_preds.device)

    cls_loss = F.cross_entropy(class_preds, class_targets)
    bbox_loss = F.mse_loss(bbox_preds, bbox_targets)

    return cls_loss + bbox_loss



import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def evaluate_map50(encoder, head, loader, device):
    encoder.eval()
    head.eval()

    metric = MeanAveragePrecision(iou_thresholds=[0.5]).to(device)

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            # Extract encoder features (CLS token)
            features_dict = encoder.forward_features(images)  # returns a dict
            features = features_dict["x_norm_clstoken"]
            #features = encoder.forward_features(images)[:, 0]

            class_logits, bbox_preds = head(features)

            # Convert bbox_preds to [x1, y1, x2, y2] format if necessary
            bbox_preds = xywh_to_xyxy(bbox_preds)

            preds_list, targets_list = [], []

            # Prepare predictions and targets per batch item
            for i in range(len(images)):
                preds_list.append({
                    "boxes": bbox_preds[i].unsqueeze(0),  # (1,4)
                    "scores": torch.softmax(class_logits[i], dim=0).max().unsqueeze(0),
                    "labels": torch.argmax(class_logits[i]).unsqueeze(0),
                })

                # Assuming targets[i] has bounding boxes and labels in COCO format
                boxes, labels = extract_boxes_labels(targets[i])
                targets_list.append({
                    "boxes": boxes.to(device),
                    "labels": labels.to(device),
                })
            metric.update(preds_list, targets_list)

    map_results = metric.compute()
    return map_results["map_50"].item()


def run_linear_probe_all_RFDETR(ssl_data_path, save_folder, current_SSL_RunID):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_dir = Path(Path(ssl_data_path).parent) / "Supervised" / "Train" / "images"
    #annotations = "./Checkpoints/" + current_SSL_RunID + "/pure_patch_class_map_balanced.json"
    annotations = "./Checkpoints/" + "run_20250618_101815" + "/pure_patch_class_map_balanced.json"
    #checkpoints = sorted(Path(save_folder).glob("*.pt"))
    checkpoints = sorted(Path("./Checkpoints/run_20250618_101815/").glob("*.pt"))

    experiment_id = setup_mlflow_experiment("Linear Probing for SSL Encoder Selection")
    run_name = f"LinearProbing_{current_SSL_RunID}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = []

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        for ckpt in checkpoints:
            encoder = vit_base(patch_size=16, img_size=224).to(device)
            state_dict = torch.load(ckpt, map_location=device)['model']
            encoder.load_state_dict(state_dict, strict=False)
            encoder.requires_grad_(False)

            detection_head = SimpleRFDETRHead(embed_dim=encoder.embed_dim, num_classes=4).to(device)
            optimizer = torch.optim.Adam(detection_head.parameters(), lr=1e-3)

            train_loader = setup_coco_loader(img_dir, annotations, batch_size=32)

            epochs = 10
            encoder.eval()
            for epoch in range(epochs):
                detection_head.train()
                for images, targets in train_loader:
                    optimizer.zero_grad()
                    images = images.to(device)
                    features_dict = encoder.forward_features(images)  # returns a dict
                    features = features_dict["x_norm_clstoken"]
                    #features = encoder.forward_features(images)[:, 0]
                    preds = detection_head(features)
                    loss = compute_detection_loss(preds, targets)
                    loss.backward()
                    optimizer.step()

            val_loader = setup_coco_loader(img_dir, annotations, batch_size=32)
            map50 = evaluate_map50(encoder, detection_head, val_loader, device)
            mlflow.log_metric(f"mAP50_{ckpt.stem}", map50)
            results.append({"checkpoint": str(ckpt), "mAP50": map50})

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
