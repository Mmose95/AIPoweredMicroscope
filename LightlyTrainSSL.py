import os, glob, os.path as op

import lightly_train

from Lightly_helpers.eta_callback import ETACallback


def _detect_user_base():
    aau = glob.glob("/work/Member Files:*")
    if aau:
        return op.basename(aau[0])        # e.g. "Member Files: MatiasMose#8097"
    sdu = [d for d in glob.glob("/work/*#*") if op.isdir(d)]
    return op.basename(sdu[0]) if sdu else None

USER_BASE_DIR = os.environ.get("USER_BASE_DIR") or _detect_user_base()
if not USER_BASE_DIR:
    raise RuntimeError("Could not determine USER_BASE_DIR")
os.environ["USER_BASE_DIR"] = USER_BASE_DIR  # make it available to child processes too

print("USER_BASE_DIR =", USER_BASE_DIR)

# If you set SSL_TRAINING_DATA to a folder path => Phase 1 (SSL) will run and produce an encoder.
# If you leave SSL_TRAINING_DATA empty/None => we assume SSL already trained and you must set SSL_ENCODER_PATH.

#CBD DATASET
#SSL_TRAINING_DATA = os.getenv("SSL_TRAINING_DATA", "/work/" + USER_BASE_DIR + "/Clinical Bacteria Dataset/DetectionDataSet/SSL").strip() or None

#Own Dataset
SSL_TRAINING_DATA = os.getenv("SSL_TRAINING_DATA", "/work/" + USER_BASE_DIR + "/CellScanData/Zoom10x - Quality Assessment/Self-Supervised").strip() or None


# LightlyTrainSSL_ucloud.py
from lightly_train import train as lightly_train
import torch, os

if __name__ == "__main__":
    # Use PyTorch's scaled-dot-product attention (fast; no Triton needed)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)

    # Be polite to NCCL on clusters
    os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    # Optional: prevent too-large host allocs
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    lightly_train(
        out="./outputLightly",          # TIP: node-local NVMe on UCloud
        data=SSL_TRAINING_DATA,               # rsync your data here in the job prolog
        model="dinov2_vit/vitb14",
        method="dinov2",

        # --- multi-GPU ---
        accelerator="gpu",
        devices=8,
        strategy="auto",

        # --- memory/perf ---
        precision="bf16-mixed",
        batch_size=16,                               # per GPU; try 32 if it fits
        num_workers=6,                               # per process; 4–8 is usually good
        loader_args=dict(
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2
        ),

        # --- transforms (multiples of 14 for vitb14) ---
        transform_args=dict(
            image_size=[224, 224],                   # 224 is 16×14
            local_view=dict(
                num_views=2,
                view_size=[98, 98],                  # 7×14
                random_resize=dict(min_scale=0.05, max_scale=0.32),
                gaussian_blur=dict(prob=0.5, blur_limit=0, sigmas=[0.1, 2.0]),
            ),
        ),

        # --- checkpoints & exports ---
        callbacks=dict(
            model_checkpoint=dict(
                monitor="train_loss", mode="min",
                save_top_k=5, save_last=True,
                every_n_train_steps=2500,            # periodic snapshots
                filename="step{step:06d}-loss{train_loss:.4f}",
            ),
            model_export=dict(every_n_epochs=10),    # periodic clean student backbone exports
        ),

        # Optional: force exact budget instead of "auto"
        # trainer_args=dict(max_steps=125000),
        # epochs=2273,

        overwrite=True,
        seed=0,
    )


