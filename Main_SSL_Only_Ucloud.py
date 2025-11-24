
from pathlib import Path
import os, glob, os.path as op

from MainPhase_QualityAssessment.Main_QualityAssessment_SSL_DINOV2 import (
    qualityAssessment_SSL_DINOV2,
)

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

WORK_ROOT = Path("/work") / USER_BASE_DIR

# This is the same root as supervised:
SSL_TRAINING_ROOT = WORK_ROOT / "CellScanData" / "Zoom10x - Quality Assessment_Cleaned"

print("USER_BASE_DIR:", USER_BASE_DIR)
print("SSL_TRAINING_ROOT:", SSL_TRAINING_ROOT)

# trackExperiment=True so you still get MLflow logging
encoder_path = qualityAssessment_SSL_DINOV2(
    True,
    SSL_TRAINING_ROOT.as_posix(),
    USER_BASE_DIR,
)

print("SSL encoder saved at:", encoder_path)
