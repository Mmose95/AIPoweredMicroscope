import os, glob, os.path as op

import lightly_train

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


if __name__ == "__main__":
    lightly_train.train(
        out="outputLightly",
        data=SSL_TRAINING_DATA,
        model="dinov2_vit/vitb14",
        method="dinov2",
    )