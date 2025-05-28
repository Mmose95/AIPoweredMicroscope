import os
import time
from types import SimpleNamespace

import torch
import mlflow
from datetime import datetime
from torch.utils.data import DataLoader

from DINO.config.DINO.DINO_5Scale_Custom import get_default_dino_args
from DINO.datasets import transforms as dino_transforms
from DINO.models import build_dino
from DINO.engine import train_one_epoch, evaluate
from DINO.models.dino.position_encoding import PositionEmbeddingSine
from Helpers_General.Supervised_learning_helpers.DINOBackboneWrapper import DINOBackboneWrapper
from Helpers_General.collate_fn import detection_collate_fn
from Utils_MLFLOW import setup_mlflow_experiment
from Helpers_General.Supervised_learning_helpers.DINOV2Backbone_DINONEW import DinoV2ForDINONEW
from Helpers_General.Supervised_learning_helpers.DetectionDataset import DetectionDataset
import torch.nn as nn

from dinov2.models.vision_transformer import vit_small
import copy


def qualityAssessment_supervised_DINO(trackExperiment, encoder_name, train_dataset, val_dataset):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if trackExperiment:
        experiment_id = setup_mlflow_experiment("Main Phase: Quality Assessment (Supervised) DINO")
        run_id = mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                  experiment_id=experiment_id)
        best_model_path = f"Checkpoints/ExId_{experiment_id}_{run_id.info.run_name}_BEST_dino_detector.pt"
        final_model_path = f"Checkpoints/ExId_{experiment_id}_{run_id.info.run_name}_FINAL_dino_detector.pt"

    from dinov2.models.vision_transformer import vit_small

    args = get_default_dino_args()
    args.device = device

    # ---------------- Model Setup ----------------
    # Load pretrained DINOv2 encoder (ViT)
    state_dict = torch.load(encoder_name, map_location="cpu")
    vit_model = vit_small(patch_size=16)
    vit_model.load_state_dict({k: v for k, v in state_dict.items() if "head" not in k}, strict=False)

    # Wrap into your custom projection model
    dinov2_extractor = DinoV2ForDINONEW(vit_model).to(device)

    # Create positional encoding
    positional_encoder = PositionEmbeddingSine(num_pos_feats=128, normalize=True)

    # Wrap both in the DINO-compatible backbone wrapper
    DINOV2backbone = DINOBackboneWrapper(dinov2_extractor, positional_encoder)

    model, criterion, postprocessors = build_dino(args, backbone=DINOV2backbone)
    model.to(device)
    criterion.to(device)

    # ---------------- Training Config ----------------
    batch_size = 4
    epochs = 50
    lr = 1e-4
    weight_decay = 1e-4
    num_workers = 8

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # DINO-compatible transforms
    transform = dino_transforms.Compose([
        dino_transforms.ToTensor(),
        dino_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    ])

    train_dataset = DetectionDataset(images_dir=os.path.join(train_dataset, "Images"),labels_dir=os.path.join(train_dataset, "Labels"),transform=transform)
    val_dataset = DetectionDataset(images_dir=os.path.join(val_dataset, "Images"),labels_dir=os.path.join(val_dataset, "Labels"),transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=detection_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=detection_collate_fn)

    # ---------------- Logging ----------------
    if trackExperiment:
        mlflow.log_params({
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "backbone": "DINOv2-ViT-small",
            "architecture": "DINO (detection transformer)"
        })

    os.makedirs("Checkpoints", exist_ok=True)

    best_val_loss = float("inf")
    start_time = time.time()

    # ---------------- Training Loop ----------------
    print("🚀 Starting Supervised DINO Training!")
    for epoch in range(epochs):

        if trackExperiment:
            mlflow.log_param("Training_data_path", train_dataset)
            mlflow.log_param("Training_data_path", val_dataset)


        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args
        )

        lr_scheduler.step()

        val_stats = evaluate(
            model=model,
            criterion=criterion,
            postprocessors=postprocessors,
            data_loader=val_loader,
            device=device,
            args=args
        )

        val_loss = val_stats["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"🔽 New best model saved at epoch {epoch + 1} with val loss {best_val_loss:.4f}")
            if trackExperiment:
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)
                mlflow.log_artifact(best_model_path)

        if trackExperiment:
            mlflow.log_metric("train_loss", train_stats["loss"], step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]["lr"], step=epoch)

        print(f"📉 Epoch [{epoch+1}/{epochs}], Train Loss: {train_stats['loss']:.4f}, Val Loss: {val_loss:.4f}")

    # ---------------- Save Final ----------------
    torch.save(model.state_dict(), final_model_path)
    if trackExperiment:
        mlflow.log_artifact(final_model_path)
        mlflow.log_metric("total_training_time_sec", time.time() - start_time)
        mlflow.end_run()

    print(f"✅ Final model saved at {final_model_path}")
