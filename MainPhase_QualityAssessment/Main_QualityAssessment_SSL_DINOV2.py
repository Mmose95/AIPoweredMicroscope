import json
import os
import csv
import shutil
import sys
from pathlib import Path

from Helpers_General.Self_Supervised_learning_Helpers.Linear_probing_for_SSL_Model_Selection import run_linear_probe_all
from Helpers_General.Self_Supervised_learning_Helpers.Linear_probing_for_SSL_Model_SelectionV2 import \
    run_linear_probe_all_RFDETR
from Helpers_General.Self_Supervised_learning_Helpers.Linear_probing_for_SSL_Model_SelectionV3 import \
    run_linear_probe_all_with_rfdetr
from Helpers_General.Self_Supervised_learning_Helpers.generate_balanced_patch_subset_for_probing import \
    generate_dataset_for_linear_probing

import torch.nn.functional as F
from datetime import datetime, timedelta
import time
import mlflow
from tqdm import tqdm
from Utils_MLFLOW import setup_mlflow_experiment
import torch
from torch.utils.data import DataLoader
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


# ---- time formatting helper ----
def _fmt_secs(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h {m:02d}m {s:02d}s"


# ---- robust, atomic save for model files ----
def save_safely(obj, path: Path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp.as_posix())
    # ensure bytes are flushed to disk before rename
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)  # atomic on POSIX


def qualityAssessment_SSL_DINOV2(trackExperiment_QualityAssessment_SSL, ssl_data_path, USER_BASE_DIR):
    ''''''''''''''''''''''''''''''''''''''' SSL Pretext '''''''''''''''''''''''''''''''''''
    ''' The starting point of this SSL pretext training is an already pretrained model: using the dino model'''

    ''' Broad explainer: To use the DINOV2 repro: https://github.com/facebookresearch/dinov2/tree/main - I had to bypass the distributed training setup (i.e. convert to a single gpu setup).
    Also a pretrained model was used (based on copious image datasets), and a custom dataloader was created.

    https://huggingface.co/facebook/dinov2-base/tree/main
    '''

    # ---------------------- Persistent dirs on UCloud -----------------------
    PROJECT_DIR = Path("/work/projects/myproj")
    CHECKPOINTS = PROJECT_DIR / "Checkpoints"
    MLRUNS_DIR = PROJECT_DIR / "mlruns"
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure MLflow writes to a persistent file store BEFORE creating/starting runs
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{MLRUNS_DIR.as_posix()}"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    # -----------------------------------------------------------------------

    # ---------------------- Dataset helpers -------------------------------
    from torch.utils.data import Dataset
    from PIL import Image

    from pathlib import Path
    from typing import List

    VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

    def collect_ssl_image_paths(root: Path, first_sample: int, last_sample: int) -> List[Path]:
        """
        Collect all image files for SSL from 'Sample XX' folders in [first_sample, last_sample].

        For each 'Sample N' folder:
          - If 'Patches for Sample N' exists, use images from there (preferred).
          - Otherwise, use images directly under 'Sample N'.

        Parameters
        ----------
        root : Path
            Root folder that contains 'Sample XX' subfolders
            e.g. /work/.../CellScanData/Zoom10x - Quality Assessment_Cleaned
        first_sample : int
            First sample index to include (inclusive)
        last_sample : int
            Last sample index to include (inclusive)

        Returns
        -------
        List[Path]
            List of image file paths.
        """
        root = Path(root)
        paths: List[Path] = []

        for sample_dir in sorted(root.glob("Sample *")):
            # Expect folder names like 'Sample 37'
            parts = sample_dir.name.split()
            try:
                num = int(parts[-1])
            except (ValueError, IndexError):
                continue

            if num < first_sample or num > last_sample:
                continue

            # Prefer patches if they exist
            patch_root = sample_dir / f"Patches for Sample {num}"
            search_root = patch_root if patch_root.is_dir() else sample_dir

            for p in search_root.rglob("*"):
                if p.is_file() and p.suffix.lower() in VALID_EXTS:
                    paths.append(p)

        if not paths:
            raise RuntimeError(
                f"No images found under {root} for samples {first_sample}-{last_sample}"
            )

        return paths

    class SimpleSSLImageDataset(Dataset):
        """
        Dataset that:
          - takes a list of image paths
          - loads PIL image
          - applies DataAugmentationDINO
          - returns list-of-crops, label_dummy
        """

        def __init__(self, image_paths: list[Path], transform):
            self.image_paths = list(image_paths)
            self.transform = transform

        def __len__(self) -> int:
            return len(self.image_paths)

        def __getitem__(self, idx):
            path = self.image_paths[idx]
            img = Image.open(path).convert("RGB")
            crops = self.transform(img)

            # DataAugmentationDINO may return a dict; we just want a flat list of tensors
            if isinstance(crops, dict):
                tensors = []
                for v in crops.values():
                    if torch.is_tensor(v):
                        tensors.append(v)
                    elif isinstance(v, (list, tuple)):
                        for x in v:
                            if torch.is_tensor(x):
                                tensors.append(x)
                    # ignore non-tensor metadata
                crops = tensors

            # At this point crops should be: list[Tensor] of length ncrops
            return crops, 0

    # ---------------------- Configs -----------------------
    DATA_PATH = ssl_data_path

    # training cfgs
    BATCH_SIZE = 128
    NUM_EPOCHS = 200
    LEARNING_RATE = 4e-4
    IMAGE_SIZE = 224
    accumulation_steps = 4  # to reach 4x batch size

    # model selection
    CHECKPOINT_FREQ = 1  # evaluate every N epochs
    WARMUP_EPOCHS = 5  # ignore very early epochs
    LOSS_TOLERANCE = 1.2  # ≤ 120 % of best loss so far
    TOP_K = 5  # keep at most K checkpoints
    EPOCH_DISTANCE_MIN = 10
    top_checkpoints = []
    best_loss_seen = float("inf")
    save_model_at_every_n = 50

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", DEVICE, "GPU: ", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    # -----------------------------------------------------

    # Create student and teacher models
    student = vit_base(patch_size=14, img_size=IMAGE_SIZE)
    student.num_channels = student.embed_dim
    teacher = deepcopy(student)
    for param in teacher.parameters():
        param.requires_grad = False

    # Load weights
    # Use on ucloud
    state_dict = torch.load(
        f"/work/{USER_BASE_DIR}/Checkpoints/Pretrained_Models/VIT_BASE_DINOV2.bin",
        map_location="cpu"
    )  # VIT BASE MODEL

    student.load_state_dict(state_dict, strict=False)
    teacher.load_state_dict(state_dict, strict=False)

    student.to(DEVICE)
    teacher.to(DEVICE)

    # Optimizer and AMP scaler
    optimizer = torch.optim.AdamW(student.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    scaler = GradScaler()

    # Loss function
    dino_loss = CustomDINOLoss(
        out_dim=768,
        ncrops=10,
        warmup_teacher_temp=0.04,
        teacher_temp=0.07,
        warmup_teacher_temp_epochs=round(NUM_EPOCHS * 0.3),
        nepochs=NUM_EPOCHS,
    ).to(DEVICE)

    # ---------------------- Transforms + Dataset / loader ------------------
    # Use env vars to define which samples to include; defaults 37–110
    if "SSL_FIRST_SAMPLE" not in os.environ or "SSL_LAST_SAMPLE" not in os.environ:
        raise RuntimeError(
            "Environment variables SSL_FIRST_SAMPLE and SSL_LAST_SAMPLE must be set.\n"
            "Example:\n"
            "   os.environ['SSL_FIRST_SAMPLE'] = '37'\n"
            "   os.environ['SSL_LAST_SAMPLE'] = '110'\n"
        )

    SSL_FIRST_SAMPLE = int(os.environ["SSL_FIRST_SAMPLE"])
    SSL_LAST_SAMPLE = int(os.environ["SSL_LAST_SAMPLE"])

    print(f"[SSL] Using sample range explicitly: {SSL_FIRST_SAMPLE}–{SSL_LAST_SAMPLE}")

    all_image_paths = collect_ssl_image_paths(
        Path(DATA_PATH),
        SSL_FIRST_SAMPLE,
        SSL_LAST_SAMPLE,
    )

    transform = DataAugmentationDINO(
        global_crops_scale=(0.2, 1.0),
        local_crops_scale=(0.05, 0.35),
        local_crops_number=8,
    )

    dataset = SimpleSSLImageDataset(all_image_paths, transform=transform)

    def ssl_collate(batch):
        """
        batch: list of (crops, label) pairs
          - crops: list[Tensor] of length ncrops
        Returns:
          - images: list[ list[Tensor] ], len(images) == batch_size
          - labels: Tensor of shape (batch_size,)
        """
        crops_list, labels = zip(*batch)  # tuples length B
        return list(crops_list), torch.tensor(labels, dtype=torch.long)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=ssl_collate,  # <- important
    )

    first_images, _ = next(iter(dataloader))
    print("[DEBUG] type(first_images):", type(first_images))
    print("[DEBUG] len(first_images):", len(first_images))
    print("[DEBUG] type(first_images[0]):", type(first_images[0]))
    print("[DEBUG] len(first_images[0]):", len(first_images[0]))
    print("[DEBUG] type(first_images[0][0]):", type(first_images[0][0]))

    print(f"[SSL] Loaded {len(all_image_paths)} images from samples {SSL_FIRST_SAMPLE}-{SSL_LAST_SAMPLE}")

    # EMA update
    def update_teacher(student_model, teacher_model, momentum):
        for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
            teacher_param.data = momentum * teacher_param.data + (1.0 - momentum) * student_param.data

    def compute_entropy(p):
        # p: (B, C) probabilities
        return -(p * (p + 1e-6).log()).sum(dim=-1).mean()

    # ---------------------- Run + folders -----------------------
    if trackExperiment_QualityAssessment_SSL:
        # Start mlflow experiment tracking (uses file store set above)
        experiment_id = setup_mlflow_experiment("Main Phase: Quality Assessment (SelfSupervised)")
        runId = mlflow.start_run(
            run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            experiment_id=experiment_id,
        )
        run_name = runId.info.run_name
        print("Experiment tracking via MLFLOW is ON!!:  ID:", experiment_id, "| run:", run_name)
    else:
        runId = None
        experiment_id = "no-mlflow"
        run_name = f"local_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print("Experiment tracking via MLFLOW is OFF!!")

    save_folder = CHECKPOINTS / run_name
    save_folder.mkdir(parents=True, exist_ok=True)

    # Small manifest to capture context
    manifest = {
        "when": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "data_path": DATA_PATH,
        "project_dir": PROJECT_DIR.as_posix(),
        "mlflow_store": MLRUNS_DIR.as_posix(),
        "save_folder": save_folder.as_posix(),
        "experiment_id": experiment_id,
        "run_name": run_name,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "image_size": IMAGE_SIZE,
        "learning_rate": LEARNING_RATE,
        "accumulation_steps": accumulation_steps,
    }
    with open(save_folder / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    # ------------------------------------------------------------

    # logging the data list (relative to DATA_PATH)
    image_folder = save_folder / "SSL"
    image_folder.mkdir(parents=True, exist_ok=True)
    output_json = image_folder / "SSLdata_filenames.json"

    root_path = Path(DATA_PATH)
    rel_files = [p.relative_to(root_path).as_posix() for p in all_image_paths]
    with open(output_json, "w") as f:
        json.dump(rel_files, f, indent=2)

    if trackExperiment_QualityAssessment_SSL:
        mlflow.log_param("number_of_images_used_in_training", len(all_image_paths))

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
        mlflow.log_param("student_num_trainable_parameters_M", num_student_params / 1e6)
        mlflow.log_param("teacher_num_trainable_parameters_M", num_teacher_params / 1e6)
        mlflow.log_param("student_total_parameters_M", num_student_params / 1e6)
        mlflow.log_param("teacher_total_parameters_M", num_teacher_params / 1e6)

        # Log data augmentation parameters
        mlflow.log_param("augmentation_global_crops_scale", transform.global_crops_scale)
        mlflow.log_param("augmentation_local_crops_scale", transform.local_crops_scale)
        mlflow.log_param("augmentation_local_crops_number", transform.local_crops_number)

        # Log data path(s)
        mlflow.log_param("Training_data_path", ssl_data_path)

        # Start time
        mlflow.set_tag("run_start_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Training setup stuff:
    best_loss = float("inf")
    best_model_path = save_folder / "BEST_dinov2_selfsup_trained.pt"
    final_model_path = save_folder / "FINAL_dinov2_selfsup_trained.pt"

    # Epoch CSV for quick inspection without MLflow UI
    epoch_csv = save_folder / "epoch_metrics.csv"
    if not epoch_csv.exists():
        with open(epoch_csv, "w", newline="") as fcsv:
            csv.writer(fcsv).writerow([
                "epoch", "avg_loss", "epoch_time_sec", "elapsed_time_sec", "eta_sec",
                "images_per_sec_epoch", "learning_rate",
                "gpu_mem_alloc_MB", "gpu_mem_reserved_MB"
            ])

    os.makedirs("Checkpoints", exist_ok=True)  # backward-compat folder (not used for saves now)

    # ---- timing/ETA bookkeeping ----
    train_t0 = time.perf_counter()
    epoch_times = []
    ema_epoch_time = None
    ALPHA = 0.3  # EMA smoothing

    print("Self-supervised training started...")

    # Actual training loop
    for epoch in range(NUM_EPOCHS):
        epoch_t0 = time.perf_counter()
        student.train()
        total_loss = 0.0
        num_accumulated_steps = 0
        optimizer.zero_grad()
        samples_this_epoch = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            with autocast():
                num_global_crops = dino_loss.ncrops - transform.local_crops_number
                num_local_crops = transform.local_crops_number
                total_crops = num_global_crops + num_local_crops

                batch_size = len(images)
                samples_this_epoch += batch_size

                # Prepare global and local crops
                global_crops = torch.cat([
                    torch.stack([crops[i].to(DEVICE, non_blocking=True)
                                 for i in range(num_global_crops)])
                    for crops in images
                ])
                local_crops = torch.cat([
                    torch.stack([crops[i].to(DEVICE, non_blocking=True)
                                 for i in range(num_global_crops, total_crops)])
                    for crops in images
                ])
                all_crops = torch.cat([global_crops, local_crops])

                student_output = student(all_crops)

                with torch.no_grad():
                    teacher_output = teacher(global_crops)

                # Extract just the global part from student_output
                student_global_output = student_output[:batch_size * num_global_crops]

                # Compute entropy
                student_probs = F.softmax(student_global_output, dim=-1)
                teacher_probs = F.softmax(teacher_output, dim=-1)

                student_entropy = compute_entropy(student_probs)
                teacher_entropy = compute_entropy(teacher_probs)

                # Chunk outputs
                student_global_chunks = student_output[:batch_size * num_global_crops].chunk(num_global_crops)
                student_local_chunks = student_output[batch_size * num_global_crops:].chunk(num_local_crops)
                student_chunks = student_global_chunks + student_local_chunks
                teacher_chunks = teacher_output.chunk(num_global_crops)

                # Compute loss
                loss = dino_loss(student_chunks, teacher_chunks) / accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                total_loss += loss.item() * accumulation_steps
                num_accumulated_steps += 1

            momentum = get_dinov2_teacher_momentum(epoch, NUM_EPOCHS)
            update_teacher(student, teacher, momentum)

            if batch_idx % 1 == 0:
                print(
                    f"\n[Epoch {epoch + 1}/ {NUM_EPOCHS}] Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}"
                )
                if trackExperiment_QualityAssessment_SSL and torch.cuda.is_available():
                    mlflow.log_metric("gpu_memory_allocated_MB", torch.cuda.memory_allocated() / 1024 ** 2, step=epoch)
                    mlflow.log_metric("gpu_memory_reserved_MB", torch.cuda.memory_reserved() / 1024 ** 2, step=epoch)
                    mlflow.log_metric("Loss-Batch", loss.item(), step=epoch * len(dataloader) + batch_idx)
                    mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)
                    mlflow.log_metric("entropy_student", student_entropy.item(),
                                      step=epoch * len(dataloader) + batch_idx)
                    mlflow.log_metric("entropy_teacher", teacher_entropy.item(),
                                      step=epoch * len(dataloader) + batch_idx)

        avg_loss = total_loss / max(1, (batch_idx + 1) // accumulation_steps)  # Average epoch loss
        if trackExperiment_QualityAssessment_SSL:
            mlflow.log_metric("Loss-Epoch", avg_loss, step=epoch)

        # ---- per-epoch timing + ETA ----
        epoch_sec = time.perf_counter() - epoch_t0
        epoch_times.append(epoch_sec)
        ema_epoch_time = epoch_sec if ema_epoch_time is None else (ALPHA * epoch_sec + (1 - ALPHA) * ema_epoch_time)

        elapsed_sec = time.perf_counter() - train_t0
        remaining_epochs = NUM_EPOCHS - (epoch + 1)
        eta_sec = ema_epoch_time * remaining_epochs
        eta_finish = datetime.now() + timedelta(seconds=eta_sec)
        img_per_sec = samples_this_epoch / epoch_sec if epoch_sec > 0 else 0.0

        # ---- per-epoch CSV append + MLflow metrics ----
        lr_now = scheduler.get_last_lr()[0]
        gpu_alloc = torch.cuda.memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0.0
        gpu_resv = torch.cuda.memory_reserved() / 1024 ** 2 if torch.cuda.is_available() else 0.0

        with open(epoch_csv, "a", newline="") as fcsv:
            csv.writer(fcsv).writerow([
                epoch + 1, avg_loss, epoch_sec, elapsed_sec, eta_sec,
                img_per_sec, lr_now, gpu_alloc, gpu_resv
            ])

        if trackExperiment_QualityAssessment_SSL:
            mlflow.log_metric("epoch_time_sec", epoch_sec, step=epoch + 1)
            mlflow.log_metric("elapsed_time_sec", elapsed_sec, step=epoch + 1)
            mlflow.log_metric("eta_sec", eta_sec, step=epoch + 1)
            mlflow.log_metric("images_per_sec_epoch", img_per_sec, step=epoch + 1)
            mlflow.log_metric("learning_rate_epoch", lr_now, step=epoch + 1)
            mlflow.log_metric("gpu_memory_allocated_MB_epoch", gpu_alloc, step=epoch + 1)
            mlflow.log_metric("gpu_memory_reserved_MB_epoch", gpu_resv, step=epoch + 1)
            mlflow.set_tag("eta_finish_localtime", eta_finish.strftime("%Y-%m-%d %H:%M:%S"))

        print(
            f"⏱ Epoch {epoch + 1}/{NUM_EPOCHS} | time {_fmt_secs(epoch_sec)} | "
            f"elapsed {_fmt_secs(elapsed_sec)} | ETA {_fmt_secs(eta_sec)} (~{eta_finish:%Y-%m-%d %H:%M})"
        )

        student_path = save_folder / f"epoch_{epoch + 1}_student.pt"
        teacher_path = save_folder / f"epoch_{epoch + 1}_teacher.pt"

        # ── smart-checkpoint logic ──────────────────────────────────────────
        entropy_gap = abs(student_entropy.item() - teacher_entropy.item())

        should_consider = (
            epoch >= WARMUP_EPOCHS
            and (epoch + 1) % CHECKPOINT_FREQ == 0
            and avg_loss <= best_loss_seen * LOSS_TOLERANCE
        )

        if should_consider:
            # ---------- maintain Top-K -------------
            if len(top_checkpoints) < TOP_K:
                save_safely({'model': student.state_dict()}, student_path)
                save_safely({'model': teacher.state_dict()}, teacher_path)
                top_checkpoints.append((epoch + 1, avg_loss, student_path.as_posix(), teacher_path.as_posix()))
                top_checkpoints.sort(key=lambda x: x[1])  # by loss
            else:
                worst_idx = max(range(TOP_K), key=lambda i: top_checkpoints[i][1])
                worst_loss = top_checkpoints[worst_idx][1]
                epoch_diffs = [abs((epoch + 1) - ep) for ep, *_ in top_checkpoints]

                if avg_loss < worst_loss and all(d >= EPOCH_DISTANCE_MIN for d in epoch_diffs):
                    _, _, w_stu, w_tea = top_checkpoints[worst_idx]
                    for p in [w_stu, w_tea]:
                        if os.path.exists(p):
                            os.remove(p)

                    save_safely({'model': student.state_dict()}, student_path)
                    save_safely({'model': teacher.state_dict()}, teacher_path)
                    top_checkpoints[worst_idx] = (epoch + 1, avg_loss, student_path.as_posix(), teacher_path.as_posix())
                    top_checkpoints.sort(key=lambda x: x[1])

            if trackExperiment_QualityAssessment_SSL:
                mlflow.log_metric("entropy_gap", entropy_gap, step=epoch)
                mlflow.log_metric("ckpt_avg_loss", avg_loss, step=epoch)
                mlflow.log_artifact(student_path.as_posix())
                mlflow.log_artifact(teacher_path.as_posix())
        # ────────────────────────────────────────────────────────────

        # keep global best_loss_seen for tolerance test
        best_loss_seen = min(best_loss_seen, avg_loss)

        # Saving models at a fixed cadence
        if (epoch + 1) % save_model_at_every_n == 0:
            save_safely({'model': student.state_dict()}, student_path)
            save_safely({'model': teacher.state_dict()}, teacher_path)
            print(f"💾 Saved checkpoint at epoch {epoch + 1}: student + teacher")

            if trackExperiment_QualityAssessment_SSL:
                mlflow.log_artifact(student_path.as_posix())
                mlflow.log_artifact(teacher_path.as_posix())
                mlflow.log_metric("Average epoch loss", avg_loss, step=epoch)
                mlflow.log_metric("teacher_momentum", momentum, step=epoch)

        # Step the scheduler
        scheduler.step()

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_safely({'model': teacher.state_dict()}, save_folder / "BEST_TEACHER_by_trainloss.pt")

            print(f"🔽 New best model saved at epoch {epoch + 1} with loss {best_loss:.4f}")
            if trackExperiment_QualityAssessment_SSL:
                mlflow.log_metric("best_model_loss", best_loss, step=epoch)
                mlflow.log_artifact(best_model_path.as_posix())

        # Logging every epoch
        print(f"📉 Epoch [{epoch + 1}/{NUM_EPOCHS}], Average loss for the epoch: {avg_loss:.4f}")

    # Final save and end
    save_safely({'model': student.state_dict()}, final_model_path)
    print("✅ Training completed. Final model saved at:", final_model_path)

    total_time_sec = time.perf_counter() - train_t0
    if trackExperiment_QualityAssessment_SSL:
        mlflow.log_artifact(final_model_path.as_posix())
        mlflow.log_metric("training_time_sec", total_time_sec)
        # also log the epoch CSV
        mlflow.log_artifact(epoch_csv.as_posix())
        mlflow.end_run()

    # (Probing currently paused)
    # generate_dataset_for_linear_probing(ssl_data_path, save_folder)
    # best_model_path = run_linear_probe_all_with_rfdetr(ssl_data_path, save_folder, id_only)

    return best_model_path.as_posix()

