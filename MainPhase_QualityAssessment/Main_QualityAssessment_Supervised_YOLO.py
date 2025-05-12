import math
from datetime import datetime
import mlflow
import torch

from Helpers_General.Supervised_learning_helpers.dino_backbone_for_yolo import DINOv2Backbone
from dinov2.models.vision_transformer import vit_base, vit_small, vit_large
from Utils_MLFLOW import setup_mlflow_experiment
import torch.nn as nn
from ultralytics import YOLO


def qualityAssessment_supervised_YOLO(trackExperiment_QualityAssessment_supervised, encoder_name):

    if trackExperiment_QualityAssessment_supervised:
        # Start mlflow experiment tracking
        experiment_id = setup_mlflow_experiment("Main Phase: Quality Assessment (Supervised)")
        runId = mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                     experiment_id=experiment_id)


    # Compatibility stage: We need to add a Feature Pyramid network (FPN) to the SSL-trained DINOV2 model to make it compatible with a YOLO network.

    # Isolate encoder backbone (by removing projection-head, which was saved during training).
    ##load full (student / encoder model)
    SSL_Encoder_full = torch.load(encoder_name)

    # Initialize a clean backbone
    backbone = vit_small(patch_size=16)
    #backbone = vit_base(patch_size=)
    #backbone = vit_large(...)

    # Remove projection head parameters from state_dict
    filtered_state = {k: v for k, v in SSL_Encoder_full.items() if "head" not in k}

    # Load only matching weights
    backbone.load_state_dict(filtered_state, strict=False)

    # Insert '_backbone' before 'ExId'
    backbone_encoder_name = encoder_name.replace('ExId', 'backbone_ExId')

    torch.save(backbone.state_dict(), backbone_encoder_name)

    with torch.cuda.device(0):
        model = YOLO("MainPhase_QualityAssessment/dino_yoloV8.yaml")

    model.train(
        data="MainPhase_QualityAssessment/Supervised_Data_YOLO.yaml",
        imgsz=384,
        epochs=100,
        batch=16,
        device=0,  # Change to 'cpu' if needed
        optimizer="Adam",  # Or "SGD"
        lr0=1e-3,
        warmup_epochs=3,
        pretrained=False  # Very important: do NOT load YOLO pretrained weights
    )

    model.save("yolo_dino_final.pt")