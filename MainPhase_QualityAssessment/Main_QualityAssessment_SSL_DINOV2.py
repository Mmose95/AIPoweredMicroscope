import json
import os
import shutil
import sys
from pathlib import Path

from Helpers_General.Self_Supervised_learning_Helpers.Linear_probing_for_SSL_Model_Selection import run_linear_probe_all
from Helpers_General.Self_Supervised_learning_Helpers.generate_balanced_patch_subset_for_probing import generate_dataset_for_linear_probing
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
from dinov2.models.vision_transformer import vit_base, vit_small
from dinov2.data import DataAugmentationDINO
from torch.cuda.amp import GradScaler, autocast
from copy import deepcopy
import warnings
import logging
from Helpers_General.Self_Supervised_learning_Helpers.momentumScheduler import get_dinov2_teacher_momentum

# Suppress irrelevant warnings
warnings.filterwarnings("ignore", message=".*xFormers is available.*")
warnings.filterwarnings("ignore", message=".*No module named 'triton'.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
logging.getLogger().setLevel(logging.ERROR)

os.environ["DINO_SINGLE_PROCESS"] = "1"

def qualityAssessment_SSL_DINOV2(trackExperiment_QualityAssessment_SSL, ssl_data_path):

    ''''''''''''''''''''''''''''''''''''''' SSL Pretext '''''''''''''''''''''''''''''''''''
    ''' The starting point of this SSL pretext training is an already pretrained model: using the dino model'''

    ''' Broad explainer: To use the DINOV2 repro: https://github.com/facebookresearch/dinov2/tree/main - I had to bypass the distributed training setup (i.e. convert to a single gpu setup).
    Also a pretrained model was used (based on copious image datasets), and a custom dataloader was created.
    
    https://huggingface.co/facebook/dinov2-base/tree/main
    '''


    # ---------------------- Config -----------------------
    DATA_PATH = ssl_data_path
    BATCH_SIZE = 128
    NUM_EPOCHS = 1 #300
    LEARNING_RATE = 2e-4
    IMAGE_SIZE = 224
    save_model_at_every_n = 50 #how often we save the student and teacher model (used in linear probing to select the best model for downstream task)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE, "GPU: ", torch.cuda.get_device_name())
    # -----------------------------------------------------

    # Create student and teacher models
    student = vit_base(patch_size=16, img_size=IMAGE_SIZE)
    student.num_channels = student.embed_dim
    teacher = deepcopy(student)
    for param in teacher.parameters():
        param.requires_grad = False

    # Load weights
    state_dict = torch.load("./Checkpoints/Pretrained_Models/VIT_BASE_DINOV2.bin", map_location="cpu") #VIT BASE MODEL
    #state_dict = torch.load("./Checkpoints/Pretrained_Models/VIT_SMALL_DINOV2.bin", map_location="cpu") #VIT SMALL MODEL
    student.load_state_dict(state_dict, strict=False)
    teacher.load_state_dict(state_dict, strict=False)

    student.to(DEVICE)
    teacher.to(DEVICE)

    # Optimizer and AMP scaler
    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    scaler = GradScaler()

    # Loss function
    dino_loss = CustomDINOLoss(
        out_dim=768,
        ncrops=10,
        warmup_teacher_temp=0.04,
        teacher_temp=0.07,
        warmup_teacher_temp_epochs=round(NUM_EPOCHS*0.3),
        nepochs=NUM_EPOCHS,
    ).to(DEVICE)

    # Transforms
    transform = DataAugmentationDINO(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8
    )

    # Use your new dataset class
    dataset = SSLImageDataset(DATA_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # EMA update
    def update_teacher(student_model, teacher_model, momentum):
        for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
            teacher_param.data = momentum * teacher_param.data + (1.0 - momentum) * student_param.data
    #####END OF UPDATE_TEACHER(): You have confused this enough times....

    if trackExperiment_QualityAssessment_SSL:
        # Start mlflow experiment tracking
        experiment_id = setup_mlflow_experiment("Main Phase: Quality Assessment (SelfSupervised)")
        runId = mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}", experiment_id=experiment_id)

        save_folder = "./Checkpoints/" + runId.info.run_name
        id_only = runId.info.run_name
        os.makedirs(save_folder, exist_ok=True)

        #logging the data:
        image_folder = Path(save_folder + "/SSL")
        os.makedirs(image_folder, exist_ok=True)
        output_json = image_folder / "SSLdata_filenames.json"

        jpg_files = sorted([f.name for f in Path(ssl_data_path + "/images").glob("*.jpg")])
        with open(output_json, "w") as f:
            json.dump(jpg_files, f, indent=2)

        print("Experiment tracking via MLFLOW is ON!!:  ID:", experiment_id)

        #to save the number of images used in this training we just count the number of labels/images
        label_dir = Path(ssl_data_path) / "labels"
        n_labels = len(list(label_dir.glob("*.txt"))) #This just counts the original label folder since it has 1 txt file pr label.
        mlflow.log_param("number of images used in training", str(n_labels))


        # Log hyperparameters dynamically
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("image_size", IMAGE_SIZE)


        # Log dynamic loss configuration
        mlflow.log_param("dino_loss_out_dim", dino_loss.out_dim)
        mlflow.log_param("dino_loss_ncrops", dino_loss.ncrops)
        mlflow.log_param("dino_loss_warmup_teacher_temp", dino_loss.warmup_teacher_temp)
        mlflow.log_param("dino_loss_teacher_temp", dino_loss.teacher_temp)
        mlflow.log_param("dino_loss_warmup_teacher_temp_epochs", dino_loss.warmup_teacher_temp_epochs)

        # Log model-specific configurations
        num_student_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
        num_teacher_params = sum(p.numel() for p in teacher.parameters())
        mlflow.log_param("student_num_trainable_parameters Millions", num_student_params/1e6)
        mlflow.log_param("teacher_num_trainable_parameters Millions", num_teacher_params/1e6)
        mlflow.log_param("student_total_parameters in millions", num_student_params/1e6)
        mlflow.log_param("teacher_total_parameters in millions", num_teacher_params/1e6)

        # Log data augmentation parameters
        mlflow.log_param("augmentation_global_crops_scale", transform.global_crops_scale)
        mlflow.log_param("augmentation_local_crops_scale", transform.local_crops_scale)
        mlflow.log_param("augmentation_local_crops_number", transform.local_crops_number)

        # Log data path(s)
        mlflow.log_param("Training_data_path", ssl_data_path)

        # Start time
        mlflow.set_tag("run_start_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    else:
        print("Experiment tracking via MLFLOW is OFF!!")
        SAVE_PATH = "dinov2_base_selfsup_trained.pt"



    #Training setup stuff:
    best_loss = float("inf")
    best_model_path = f"{save_folder}/ExId_{experiment_id}_{runId.info.run_name}_BEST_dinov2_selfsup_trained.pt"
    #final_model_path = f"{save_folder}/ExId_{experiment_id}_{runId.info.run_name}_FINAL_dinov2_selfsup_trained.pt"
    os.makedirs("Checkpoints", exist_ok=True)

    start_time = time.time() #Lets time this sucker
    print("LET THE TRAINING COMMENCE!!!!!")
    # Actual training loop

    for epoch in range(NUM_EPOCHS):
        student.train()
        total_loss = 0.0

        for batch_idx, (images, _) in enumerate(dataloader):
            num_global_crops = dino_loss.ncrops - transform.local_crops_number
            num_local_crops = transform.local_crops_number
            total_crops = num_global_crops + num_local_crops

            batch_size = len(images)

            with autocast():
                # Student processes ALL crops (global + local):
                # First stack all global crops, then local crops separately
                global_crops = torch.cat([torch.stack([crops[i].to(DEVICE, non_blocking=True)
                                                       for i in range(num_global_crops)])
                                          for crops in images])  # (batch_size*num_global_crops, C, H, W)

                local_crops = torch.cat([torch.stack([crops[i].to(DEVICE, non_blocking=True)
                                                      for i in range(num_global_crops, total_crops)])
                                         for crops in images])  # (batch_size*num_local_crops, C, H, W)

                # Concatenate global + local explicitly
                all_crops = torch.cat([global_crops, local_crops])  # (batch_size*(global+local), C, H, W)

                # Pass through student
                student_output = student(all_crops)

                # Teacher ONLY sees global crops
                with torch.no_grad():
                    teacher_output = teacher(global_crops)

                # Now explicitly chunk student output:
                student_global_chunks = student_output[:batch_size * num_global_crops].chunk(num_global_crops)
                student_local_chunks = student_output[batch_size * num_global_crops:].chunk(num_local_crops)
                student_chunks = student_global_chunks + student_local_chunks

                # Teacher outputs are straightforward:
                teacher_chunks = teacher_output.chunk(num_global_crops)

                # Compute loss with explicitly structured chunks:
                loss = dino_loss(student_chunks, teacher_chunks)
                total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            momentum = get_dinov2_teacher_momentum(epoch, NUM_EPOCHS)
            update_teacher(student, teacher, momentum)

            if batch_idx % 1 == 0:
                print(f"\n[Epoch {epoch + 1}/ {NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} - Loss: {loss.item():.4f}")
                if trackExperiment_QualityAssessment_SSL:
                    mlflow.log_metric("gpu_memory_allocated_MB", torch.cuda.memory_allocated() / 1024 ** 2, step=epoch)
                    mlflow.log_metric("gpu_memory_reserved_MB", torch.cuda.memory_reserved() / 1024 ** 2, step=epoch)
                    mlflow.log_metric("Loss-Batch", loss.item(), step=epoch * len(dataloader) + batch_idx)
                    #mlflow.log_metric("student_param_norm", student.backbone[0].encoder.patch_embed.proj.weight.norm().item(), step=epoch)
                    #mlflow.log_metric("teacher_param_norm", teacher.backbone[0].encoder.patch_embed.proj.weight.norm().item(), step=epoch)
                    mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)

        avg_loss = total_loss / len(dataloader)

        if (epoch + 1) % save_model_at_every_n == 0:
            student_path = f"{save_folder}/epoch_{epoch + 1}_student.pt"
            teacher_path = f"{save_folder}/epoch_{epoch + 1}_teacher.pt"

            torch.save({'model': student.state_dict()}, student_path)
            torch.save({'model': teacher.state_dict()}, teacher_path)

            print(f"💾 Saved checkpoint at epoch {epoch + 1}: student + teacher")

            if trackExperiment_QualityAssessment_SSL:
                mlflow.log_artifact(student_path)
                mlflow.log_artifact(teacher_path)
                mlflow.log_metric("Average epoch loss", avg_loss, step=epoch)
                mlflow.log_metric("teacher_momentum", momentum, step=epoch)
        # Step the scheduler
        scheduler.step()

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            #torch.save({'model': student.state_dict()}, best_model_path)
            print(f"🔽 New best model saved at epoch {epoch + 1} with loss {best_loss:.4f}")
            #if trackExperiment_QualityAssessment_SSL:
                #mlflow.log_metric("best_model_loss", best_loss, step=epoch)
                #mlflow.log_artifact(best_model_path)

        # Logging every epoch
        print(f"📉 Epoch [{epoch + 1}/{NUM_EPOCHS}], Average loss for the epoch Loss: {avg_loss:.4f}")

    # Final save and end
    # Save final model
    #torch.save({'model': student.state_dict()}, final_model_path)
    #print("✅ Training completed. Final model saved at:", final_model_path)

    if trackExperiment_QualityAssessment_SSL:
        #mlflow.log_artifact(final_model_path)
        mlflow.log_metric("training_time_sec", time.time() - start_time)
        mlflow.end_run()

    '''To select the best model for the downstram task we do a linear probing on a purified subset of our data: 
    We check all our supervised data for patches that only contain one class. These are selected and split into test, val and train to train a small linear classifier
    on all models saved during SSL training'''

    generate_dataset_for_linear_probing(ssl_data_path, save_folder)
    best_model_path = run_linear_probe_all(ssl_data_path, save_folder, id_only)


    return best_model_path
