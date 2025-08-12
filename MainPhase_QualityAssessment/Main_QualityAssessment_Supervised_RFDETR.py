import os
import shutil
from pathlib import Path
import re
import torch

from detr import RFDETRBase


def qualityAssessment_supervised_RFDETR(trackExperiment, encoder_name, supervised_data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Confirm GPU usage
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {gpu_name}")
    else:
        gpu_name = "CPU"
        print("⚠️ No GPU detected. Running on CPU.")


    # We need to wrap the student model to be accepted in the RF-framework:
    encoder_weights = torch.load(encoder_name)

    # Prefix with 'backbone.'
    wrapped_weights = {f"backbone.{k}": v for k, v in encoder_weights.items()}

    # Add dummy detection head so RF-DETR doesn't crash
    num_classes = 5
    wrapped_weights["class_embed.weight"] = torch.zeros(num_classes, 256)
    wrapped_weights["class_embed.bias"] = torch.zeros(num_classes)

    # Wrap into a RF-DETR-style checkpoint
    checkpoint = {"model": wrapped_weights}
    encoder_name_red = encoder_name.split(".")[0]

    match = re.search(r'(run_\d{8}_\d{6})', encoder_name_red)
    if match:
        run_id = match.group(1)
        base_dir = 'C:/Users/SH37YE/Desktop/PhD_Code_github/AIPoweredMicroscope/Checkpoints'
        run_folder = os.path.join(base_dir, run_id)
        # Create the folder if it doesn't exist
        os.makedirs(run_folder, exist_ok=True)
        print(f"✅ Created or found existing folder: {run_folder}")
    else:
        print("❌ run ID not found in the path.")

    #generating the correct path to save at:
    base_dir = os.path.dirname(encoder_name_red)  # .../Checkpoints
    filename = os.path.basename(encoder_name_red)  # ExId_...

    # Inject run_id as subfolder
    corrected_path = os.path.join(base_dir , filename)

    torch.save(checkpoint, corrected_path + "_converted_rfdetr_checkpoint.pth")

    #also, to track back data, we save the annotations.jason file to the folder along with the model (this contains image names which enables traceback)
    Path(base_dir + "/train").mkdir(parents=True, exist_ok=True)
    shutil.copy(supervised_data_path + "/Train/_annotations.coco.json",base_dir + "/train")
    Path(base_dir + "/val").mkdir(parents=True, exist_ok=True)
    shutil.copy(supervised_data_path + "/Valid/_annotations.coco.json",base_dir + "/val")
    Path(base_dir + "/test").mkdir(parents=True, exist_ok=True)
    shutil.copy(supervised_data_path + "/Test/_annotations.coco.json",base_dir + "/test")


    model = RFDETRBase(pretrain_weights= corrected_path + "_converted_rfdetr_checkpoint.pth")

    #note: MLFLOW tracking is done explicitly inside the model.train() function.
    model.train(
        dataset_dir=supervised_data_path,
        epochs=60,
        batch_size=8, #16
        grad_accum_steps=4,
        lr=2e-4,
        num_queries=100,
        num_workers=12,
        trackExperiment=trackExperiment,
        amp=True,
        max_dets=[1, 30],
        early_stopping=False,
        onecyclelr=False,
        persistent_workers=True,
    )

    '''#Debug run.
    model.train(
        dataset_dir=supervised_data_path,
        epochs=3,                      # 🔽 Minimal epoch
        batch_size=2,                 # 🔽 Tiny batch for quick data load
        grad_accum_steps=1,          # 🔽 No accumulation
        lr=1e-4,                      # 🔁 Leave default or reduce slightly
        num_queries=100,              # 🔽 Lower if model allows
        num_workers=8,                # 🔽 For reproducibility & simplicity
        trackExperiment=trackExperiment,
        amp=False,                    # 🔽 Disable AMP to avoid extra overhead
        max_dets=[1, 10],             # 🔽 Fewer detections = faster COCO eval
        early_stopping=False,         # 🔽 Not needed for 1 epoch
        onecyclelr=False,             # 🔽 Skip scheduler for fast test
    )'''


    #NUMBER TO BEAT IN TEST: . For both validation and test sets, the mAP 0.5 score exceeds 0.73