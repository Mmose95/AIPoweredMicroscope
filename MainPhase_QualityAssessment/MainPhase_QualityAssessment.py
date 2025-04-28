import os
import sys
from Helpers_General.warnings import suppress_stdout_stderr
#sys.stdout = open(os.devnull, 'w')
#sys.stderr = open(os.devnull, 'w')

with suppress_stdout_stderr():
    import torch
    from dinov2.models.vision_transformer import vit_base
    from dinov2.loss.dino_clstoken_loss import DINOLoss
    import timm


from datetime import datetime
import time
import mlflow
from tqdm import tqdm
from Utils_MLFLOW import setup_mlflow_experiment
import torch
from torch.utils.data import DataLoader
from dinov2.CustomDataset import SSLImageDataset
from dinov2.loss.CustomDINO import CustomDINOLoss
from dinov2.models.vision_transformer import vit_base
from dinov2.data import DataAugmentationDINO
from torch.cuda.amp import GradScaler, autocast
from copy import deepcopy
import warnings
import logging

# Suppress irrelevant warnings
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", message=".*No module named 'triton'.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
logging.getLogger().setLevel(logging.ERROR)

os.environ["DINO_SINGLE_PROCESS"] = "1"

def qualityAssessment_SSL(trackExperiment_QualityAssessment_SSL, ssl_data_path):

    ''''''''''''''''''''''''''''''''''''''' SSL Pretext '''''''''''''''''''''''''''''''''''
    ''' The starting point of this SSL pretext training is an already pretrained model: using the dino model'''

    ''' Broad explainer: To use the DINOV2 repro: https://github.com/facebookresearch/dinov2/tree/main - I had to bypass the distributed training setup (i.e. convert to a single gpu setup).
    Also a pretrained model was used (based on copious image datasets), and a custom dataloader was created.'''


    # ---------------------- Config -----------------------
    DATA_PATH = ssl_data_path
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    IMAGE_SIZE = 384
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE, "GPU: ", torch.cuda.get_device_name())
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
    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=0.04)
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

    # Transforms
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

    if trackExperiment_QualityAssessment_SSL:
        #Setting up mlflow experimentation
        experiment_id = setup_mlflow_experiment("Main Phase: Quality Assessment (SelfSupervised)")
        mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}", experiment_id=experiment_id)
        SAVE_PATH = "/" + experiment_id + "_dinov2_base_selfsup_trained.pt"
        print("Experiment tracking via MLFLOW is ON!!:  ID:", experiment_id)

        # ───── Hyperparameters ─────
        mlflow.log_param("Hyp. Param - batch_size", BATCH_SIZE)
        mlflow.log_param("Hyp. Param - num_epochs", NUM_EPOCHS)
        mlflow.log_param("Hyp. Param - learning_rate", LEARNING_RATE)
        mlflow.log_param("Hyp. Param - image_size", IMAGE_SIZE)

        # ───── Loss Config ─────
        mlflow.log_param("DINOLOSS_cfg - out_dim", 768)
        mlflow.log_param("DINOLOSS_cfg - ncrops", 2)
        mlflow.log_param("DINOLOSS_cfg - teacher_temp", 0.07)
        mlflow.log_param("DINOLOSS_cfg - warmup_teacher_temp", 0.04)
        mlflow.log_param("DINOLOSS_cfg - warmup_teacher_temp_epochs", 10)

        # ───── Data Augmentation ─────
        mlflow.log_param("SSL_Augmentation - global_crops_scale", str((0.4, 1.0)))
        mlflow.log_param("SSL_Augmentation - local_crops_scale", str((0.05, 0.4)))
        mlflow.log_param("SSL_Augmentation - local_crops_number", 0)

        mlflow.set_tag("run_start_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    else:
        print("Experiment tracking via MLFLOW is OFF!!")
        SAVE_PATH = "dinov2_base_selfsup_trained.pt"


    #Training setup stuff:
    best_loss = float("inf")
    best_model_path = f"Checkpoints/{experiment_id}_BEST_dinov2_base_selfsup_trained.pt"
    final_model_path = f"Checkpoints/{experiment_id}_FINAL_dinov2_base_selfsup_trained.pt"
    os.makedirs("Checkpoints", exist_ok=True)

    start_time = time.time() #Lets time this sucker
    print("LET THE TRAINING COMMENCE!!!!!")
    # Actual training loop

    for epoch in range(NUM_EPOCHS):
        student.train()
        total_loss = 0.0

        for batch_idx, (images, _) in enumerate(dataloader):
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

            if batch_idx % 1 == 0:
                print(f"\n[Epoch {epoch + 1}/ {NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
                if trackExperiment_QualityAssessment_SSL:
                    mlflow.log_metric("gpu_memory_allocated_MB", torch.cuda.memory_allocated() / 1024 ** 2, step=epoch)
                    mlflow.log_metric("gpu_memory_reserved_MB", torch.cuda.memory_reserved() / 1024 ** 2, step=epoch)
                    mlflow.log_metric("Loss-Batch", loss.item(), step=epoch * len(dataloader) + batch_idx)

        avg_loss = total_loss / len(dataloader)
        mlflow.log_metric("Average epoch loss", avg_loss, step=epoch)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), best_model_path)
            print(f"🔽 New best model saved at epoch {epoch + 1} with loss {best_loss:.4f}")
            if trackExperiment_QualityAssessment_SSL:
                mlflow.log_metric("best_model_loss", best_loss, step=epoch)
                mlflow.log_artifact(best_model_path)

        # Logging every epoch
        print(f"📉 Epoch [{epoch + 1}/{NUM_EPOCHS}], Average loss for the epoch Loss: {avg_loss:.4f}")

    # Final save and end
    # Save final model
    torch.save(student.state_dict(), final_model_path)
    print("✅ Training completed. Final model saved at:", final_model_path)

    if trackExperiment_QualityAssessment_SSL:
        mlflow.log_artifact(final_model_path)
        mlflow.log_metric("training_time_sec", time.time() - start_time)
        mlflow.end_run()

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