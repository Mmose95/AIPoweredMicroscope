# Main_Orchestrator_UCloud.py
# Non-interactive orchestrator for UCloud (Linux, headless).
# - No input() / GUI dialogs
# - No Windows .cmd
# - Uses hard-coded paths below (can be overridden with env vars)
# - Keeps your original flow: if no SSL data is given, expect a ready encoder to use.

from __future__ import annotations
import os
import sys
from pathlib import Path
import multiprocessing

# ---- your imports (unchanged) ----
from Helpers_General.FullDataSplitsSafe import fullDataSplits_SampleSafe
from MainPhase_QualityAssessment.Main_QualityAssessment_SSL_DINOV2 import qualityAssessment_SSL_DINOV2
from MainPhase_QualityAssessment.Main_QualityAssessment_Supervised_RFDETR import qualityAssessment_supervised_RFDETR
# (keep other imports you actually use)

# =========================
# CONFIG (edit these)
# You can also override any of these via environment variables of the same name.
# =========================

#Determine basedir
import os, glob, os.path as op

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
#SSL_TRAINING_DATA = os.getenv("SSL_TRAINING_DATA", "/work/" + USER_BASE_DIR + "/CellScanData/Zoom10x - Quality Assessment/Self-supervised/").strip() or None
SSL_TRAINING_DATA = os.getenv("SSL_TRAINING_DATA", "/work/" + "/CellScanData/Zoom10x - Quality Assessment/Self-supervised/").strip() or None


# Supervised data path (required if you run supervised training)
SUPERVISED_TRAINING_DATA = os.getenv("SUPERVISED_TRAINING_DATA", "").strip() or None

# Path/name of a pre-trained encoder to use when NOT running SSL in this pass.
# (If you do run SSL, the encoder returned from SSL will be used automatically.)
SSL_ENCODER_PATH = os.getenv("SSL_ENCODER_PATH", "").strip() or None

# Toggle phases (default follows your original “top-level switches” idea)
GENERATE_OVERALL_SPLITS = os.getenv("GENERATE_OVERALL_SPLITS", "0") == "1"
RUN_SSL = SSL_TRAINING_DATA is not None
RUN_SUPERVISED = os.getenv("RUN_SUPERVISED", "0") == "1"   # default: run supervised

# Optional: if you want SSL → Supervised conjunction automatically when SSL runs
SSL_SUP_CONJUNCTION = os.getenv("SSL_SUP_CONJUNCTION", "0") == "1"

# Example of (commented) split generation paths if you need that phase on Linux:
# SPLIT_IMAGES_DIR = "r"C:\Users\SH37YE\Desktop\Clinical_Bacteria_DataSet\DetectionDataSet\images""
# SPLIT_LABELS_DIR = r"C:\Users\SH37YE\Desktop\Clinical_Bacteria_DataSet\DetectionDataSet\labels""
# SPLIT_ROOT_DIR = r"C:\Users\SH37YE\Desktop\Clinical_Bacteria_DataSet\DetectionDataSet"

# =========================
# Helpers
# =========================

def _exists_dir(p: str | Path) -> bool:
    return bool(p) and Path(p).is_dir()

def _exists_file(p: str | Path) -> bool:
    return bool(p) and Path(p).is_file()

def main() -> None:
    print("=== UCloud Orchestrator (non-interactive) ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"CUDA available (if torch installed): try `import torch; torch.cuda.is_available()` in a REPL.")

    # Phase: (optional) split generation — requires valid Linux paths; no Windows paths here
    if GENERATE_OVERALL_SPLITS:
        # Ensure you set the three dirs above before enabling this on UCloud.
        raise RuntimeError(
            "GENERATE_OVERALL_SPLITS=1 but no Linux paths configured in this file. "
            "Set SPLIT_* paths for your dataset before enabling."
        )
        # fullDataSplits_SampleSafe(SPLIT_IMAGES_DIR, SPLIT_LABELS_DIR, SPLIT_ROOT_DIR)

    encoder_name_or_path = SSL_ENCODER_PATH  # Will be replaced if we run SSL

    # --------------------------
    # Phase 1: SSL (if requested)
    # --------------------------
    if RUN_SSL:
        if not _exists_dir(SSL_TRAINING_DATA):
            raise FileNotFoundError(
                f"SSL_TRAINING_DATA does not exist or is not a directory: {SSL_TRAINING_DATA}"
            )
        print(f"[SSL] Starting SSL training with data: {SSL_TRAINING_DATA}")
        # Your SSL function returns the encoder identifier/name/path you need later

        encoder_name_or_path = qualityAssessment_SSL_DINOV2(True, SSL_TRAINING_DATA, USER_BASE_DIR)
        print(f"[SSL] Finished. Encoder produced: {encoder_name_or_path}")

    # ---------------------------------
    # Phase 2: Supervised (if requested)
    # ---------------------------------
    if RUN_SUPERVISED:
        # If we didn't run SSL now, we must have an existing encoder to use.
        if not RUN_SSL:
            if not _exists_file(encoder_name_or_path) and not encoder_name_or_path:
                raise FileNotFoundError(
                    "No SSL was run in this pass and SSL_ENCODER_PATH is missing/invalid. "
                    "Set SSL_ENCODER_PATH to your trained encoder file."
                )
            print(f"[SUP] Using existing encoder: {encoder_name_or_path}")
        else:
            if not SSL_SUP_CONJUNCTION:
                print("[SUP] Conjunction disabled, but SSL ran. "
                      "Provide SSL_ENCODER_PATH if you want a specific encoder different from SSL output.")
            else:
                print(f"[SUP] Conjunction enabled. Using SSL-produced encoder: {encoder_name_or_path}")

        if not _exists_dir(SUPERVISED_TRAINING_DATA):
            raise FileNotFoundError(
                f"SUPERVISED_TRAINING_DATA does not exist or is not a directory: {SUPERVISED_TRAINING_DATA}"
            )

        print(f"[SUP] Starting supervised training with data: {SUPERVISED_TRAINING_DATA}")
        qualityAssessment_supervised_RFDETR(
            trackExperiment=True,
            encoder_name=encoder_name_or_path,
            supervised_data_path=SUPERVISED_TRAINING_DATA,
        )
        print("[SUP] Finished supervised training.")

    print("✅ Done.")

if __name__ == "__main__":
    # Keeping spawn for cross-platform consistency (like your original).
    multiprocessing.set_start_method('spawn', force=True)
    main()
