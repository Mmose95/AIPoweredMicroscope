
import sys
from pathlib import Path
#from dinov2.train.train import main as train_main, get_args_parser
import mlflow
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import timm
from PIL import Image
from Helpers_General.Self_Supervised_learning_Helpers.SSL_functions import PILImageDataset, MicroscopyDataset, \
    ssl_transform
from MainPhase_QualityAssessment.SSL_DINO_TrainingLoop import DINO_training_loop
from Utils_MLFLOW import setup_mlflow_experiment
import os
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from dinov2.CustomDataset import SSLImageDataset
from dinov2.loss.CustomDINO import CustomDINOLoss
from dinov2.models.vision_transformer import vit_base
from dinov2.loss.dino_clstoken_loss import DINOLoss
from dinov2.data import DataAugmentationDINO
from torch.cuda.amp import GradScaler, autocast
from copy import deepcopy

os.environ["DINO_SINGLE_PROCESS"] = "1"

def qualityAssessment_SSL(trackExperiment_QualityAssessment_SSL, ssl_data_path):

    if trackExperiment_QualityAssessment_SSL:
        #Setting up mlflow experimentation
        experiment_id = setup_mlflow_experiment("Main Phase: Quality Assessment (SelfSupervised)")

    ''''''''''''''''''''''''''''''''''''''' SSL Pretext '''''''''''''''''''''''''''''''''''
    ''' The starting point of this SSL pretext training is an already pretrained model: using the dino model'''

    # ---------------------- Config -----------------------
    DATA_PATH = ssl_data_path  # Change this
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 384
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE, "GPU: ", torch.cuda.get_device_name())
    SAVE_PATH = "dinov2_base_selfsup_trained.pt"
    # -----------------------------------------------------

    # Create student and teacher models
    student = vit_base(patch_size=14, img_size=IMAGE_SIZE)
    teacher = deepcopy(student)
    for param in teacher.parameters():
        param.requires_grad = False

    # Load weights
    state_dict = torch.load("C:/Users/SH37YE\Desktop\PhD_Code_github\AIPoweredMicroscope\Checkpoints\pytorch_model.bin", map_location="cpu")
    student.load_state_dict(state_dict, strict=False)
    teacher.load_state_dict(state_dict, strict=False)

    student.to(DEVICE)
    teacher.to(DEVICE)

    # Optimizer and AMP scaler
    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    # Loss function
    dino_loss = CustomDINOLoss(
        out_dim=768,
        ncrops=2,
        warmup_teacher_temp=0.04,
        teacher_temp=0.07,
        warmup_teacher_temp_epochs=10,
        nepochs=NUM_EPOCHS,
    ).to(DEVICE)

    # Transforms (you already have this part correct)
    transform = DataAugmentationDINO(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=0  # Only global crops for now
    )

    # Use your new dataset class
    dataset = SSLImageDataset(DATA_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # EMA update
    def update_teacher(student_model, teacher_model, momentum=0.996):
        for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
            teacher_param.data = momentum * teacher_param.data + (1.0 - momentum) * student_param.data

    # Training loop
    for epoch in range(NUM_EPOCHS):
        student.train()
        total_loss = 0.0

        for images, _ in dataloader:
            flat_images = torch.stack([crop.to(DEVICE, non_blocking=True) for crops in images for crop in crops])
            with autocast():
                student_output = student(flat_images, masks=None)
                with torch.no_grad():
                    teacher_output = teacher(flat_images, masks=None)
                loss = dino_loss(student_output, teacher_output)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            update_teacher(student, teacher)
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(student.state_dict(), SAVE_PATH)

    print("✅ Training completed. Final model saved at:", SAVE_PATH)

    '''# Define paths
    DATASET_PATH = "ImageFolder=" + ssl_data_path  # Change this!
    OUTPUT_DIR = "./ssl_output"
    CONFIG_PATH = "dinov2/configs/train/SSL_QA_config.yaml"

    # Make sure output directory exists
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    parser = get_args_parser()

    ssl_data_path = ssl_data_path.replace("\\", "/")
    dataset_arg = f"train.dataset_path=Unlabeled={ssl_data_path}"

    args_list = [
        "--config-file", CONFIG_PATH,
        "--output-dir", OUTPUT_DIR,
        dataset_arg,
        "train.batch_size=16",
        "distributed.enabled=false",
        "distributed.backend=null"
    ]

    print("\n🚀 Launching DINOv2 training with these args:")
    for a in args_list:
        print(" -", a)

    args = parser.parse_args(args_list)

    os.environ["DINO_SINGLE_PROCESS"] = "1"


    train_main(args)


    metricSSL = "PH1_SSL"


    if trackExperiment_QualityAssessment_SSL == True:
        with mlflow.start_run(experiment_id=experiment_id) as run:

            # Log parameters and metrics
            mlflow.log_param("SSL_METRIC_TEST", metricSSL)'''


    ''''''''''''''''''''''''' Downstream task (Bounding box Classification)'''''''''''''''''''''''''''''''''
'''
    ### Setting up mlflow experimentation
    experiment_id = setup_mlflow_experiment("Main Phase: Quality Assessment (Supervised)")

    ### Full model architecture configuration.

    # Load a YOLOv8 model
    yolo_model = YOLO("yolov8s.pt")  # Load YOLOv8 Small for faster training

    # Load pretrained DINOv2
    dinov2_model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False)
    dinov2_model.load_state_dict(torch.load("dinov2_pretrained_microscopy.pth"))

    # Remove DINOv2's classification head and use it as a feature extractor
    dinov2_model.head = torch.nn.Identity()

    # Assign DINOv2 as YOLO’s new backbone
    yolo_model.model.model[0] = dinov2_model

    # Save new model config
    yolo_model.save("yolov8_with_dino.pt") #This is the actual full model architechture (YoloV8 model with (pretrained) DINOV2 backbone)

    ### Fine-tuning of full model architecture (using labeled data)


    metricSupervised = "PH1_Supervised"

    if trackExperiment_QualityAssessment_Supervised == True:
        with mlflow.start_run(experiment_id=experiment_id) as run:
            # Log parameters and metrics
            mlflow.log_param("SUPERVISED_METRIC_TEST", metricSupervised)'''