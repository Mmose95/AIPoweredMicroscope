from __future__ import annotations

import os
import runpy
from pathlib import Path


def _setdefault(name: str, value: str) -> None:
    if not os.environ.get(name, "").strip():
        os.environ[name] = value


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATASET = HERE / "Stat_Dataset" / "QA-2025v1_TwoClass_OVR_V2_20260618-101346"
TRAIN_SCRIPT = HERE / "TrainRFDETR_SOLO_SingleClass_STATIC_Ucloud.py"


def _detect_ucloud_user_base() -> str:
    work = Path("/work")
    if not work.exists():
        return ""
    member = sorted(work.glob("Member Files:*"))
    if member:
        return member[0].name
    hashed = sorted([p for p in work.glob("*#*") if p.is_dir()])
    return hashed[0].name if hashed else ""


UCLOUD_USER_BASE = _detect_ucloud_user_base()
if UCLOUD_USER_BASE:
    OUTPUT_ROOT = Path("/work") / UCLOUD_USER_BASE / "RFDETR_SOLO_OUTPUT" / "FINAL_B200"
else:
    OUTPUT_ROOT = HERE / "RFDETR_SOLO_OUTPUT" / "FINAL_B200"


# Final-study defaults. Override any of these from the shell before running.
_setdefault("RFDETR_RUNTIME_PROFILE", "ucloud" if UCLOUD_USER_BASE else "local")
_setdefault("RFDETR_EXPERIMENT_MODE", "final_b200")
_setdefault("RFDETR_HPO_TARGET", "two-class")
_setdefault("DATASET_TWO_CLASS", str(DATASET))
_setdefault("OUTPUT_ROOT", str(OUTPUT_ROOT))

# Candidate selection should not touch the clinically validated test split.
_setdefault("RFDETR_RUN_TEST", "0")

# Full-image mode. Per-row final_b200 configs request model-native full resolutions.
_setdefault("RFDETR_INPUT_MODE", "640")
_setdefault("RFDETR_PREFER_MODEL_NATIVE_DEFAULTS", "1")
_setdefault("RFDETR_ALLOW_NON_NATIVE_PRETRAIN_RESOLUTION", "0")

# B200 plan defaults.
_setdefault("MAX_PARALLEL", "8")
_setdefault("NUM_WORKERS", "8")
_setdefault("RFDETR_FINAL_B200_EPOCHS", "160")
_setdefault("RFDETR_FINAL_B200_PATIENCE", "25")
_setdefault("RFDETR_FINAL_B200_INCLUDE_LARGE", "1")
_setdefault("RFDETR_FINAL_B200_INCLUDE_2XL", "1")
_setdefault("RFDETR_OOM_MAX_RETRIES", "8")


if __name__ == "__main__":
    print("[FINAL_B200] dataset:", os.environ["DATASET_TWO_CLASS"])
    print("[FINAL_B200] output_root:", os.environ["OUTPUT_ROOT"])
    print("[FINAL_B200] experiment_mode:", os.environ["RFDETR_EXPERIMENT_MODE"])
    print("[FINAL_B200] run_test:", os.environ["RFDETR_RUN_TEST"])
    print("[FINAL_B200] max_parallel:", os.environ["MAX_PARALLEL"])
    runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
