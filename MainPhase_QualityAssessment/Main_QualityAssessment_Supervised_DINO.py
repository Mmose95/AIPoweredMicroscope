import math
from datetime import datetime
import mlflow
import torch

from DINO.models import build_dino
from Helpers_General.Supervised_learning_helpers.DINOV2_Backbone_for_DINO import DINOv2Backbone_DINO
from dinov2.models.vision_transformer import vit_base, vit_small, vit_large
from Utils_MLFLOW import setup_mlflow_experiment
import torch.nn as nn
from ultralytics import YOLO

def qualityAssessment_supervised_DINO(trackExperiment_QualityAssessment_supervised, encoder_name):

    if trackExperiment_QualityAssessment_supervised:
        # Start mlflow experiment tracking
        experiment_id = setup_mlflow_experiment("Main Phase: Quality Assessment (Supervised) DINO")
        runId = mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                     experiment_id=experiment_id)

    backbone = vit_small(patch_size=16)
    backbone.load_state_dict(torch.load(encoder_name), strict=False)
    stop = 1