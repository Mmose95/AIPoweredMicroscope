# LightlyTrainSSL.py
import os, glob, os.path as op
from lightly_train import train as lightly_train

def _detect_user_base():
    aau = glob.glob("/work/Member Files:*")
    if aau:
        return op.basename(aau[0])
    sdu = [d for d in glob.glob("/work/*#*") if op.isdir(d)]
    return op.basename(sdu[0]) if sdu else None

USER_BASE_DIR = os.environ.get("USER_BASE_DIR") or _detect_user_base()
if not USER_BASE_DIR:
    raise RuntimeError("Could not determine USER_BASE_DIR")
os.environ["USER_BASE_DIR"] = USER_BASE_DIR

SSL_TRAINING_DATA = os.getenv(
    "SSL_TRAINING_DATA",
    f"/work/{USER_BASE_DIR}/CellScanData/Zoom10x - Quality Assessment/Self-Supervised"
).strip() or None

if __name__ == "__main__":
    lightly_train(
        out="outputLightly",
        data=SSL_TRAINING_DATA,
        model="dinov2_vit/vitb14",
        method="dinov2",

        accelerator="gpu",
        devices=8,
        strategy="ddp_find_unused_parameters_false",  # ← use DDP, not spawn

        precision="bf16-mixed",
        batch_size=32,                 # per GPU
        num_workers=6,
        loader_args=dict(
            persistent_workers=True,   # fine in a script/external process
            pin_memory=True,
            prefetch_factor=2
        ),
        transform_args=dict(
            image_size=[224, 224],
            local_view=dict(
                num_views=2,
                view_size=[98, 98],
                random_resize=dict(min_scale=0.05, max_scale=0.32),
                gaussian_blur=dict(prob=0.5, blur_limit=0, sigmas=[0.1, 2.0]),
            ),
        ),
        overwrite=True,
    )
