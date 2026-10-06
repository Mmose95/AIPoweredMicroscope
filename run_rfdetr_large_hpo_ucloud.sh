#!/usr/bin/env bash
set -Eeuo pipefail

STAGE="${1:-}"
SELECTED_LR="${2:-}"
SELECTED_ENCODER_LR="${3:-}"
SELECTED_WEIGHT_DECAY="${4:-}"

if [ -z "${STAGE}" ] || [[ ! "${STAGE}" =~ ^(lr|wd|final)$ ]]; then
  echo "Usage:"
  echo "  bash run_rfdetr_large_hpo_ucloud.sh lr"
  echo "  bash run_rfdetr_large_hpo_ucloud.sh wd <selected_lr> <selected_encoder_lr>"
  echo "  bash run_rfdetr_large_hpo_ucloud.sh final <selected_lr> <selected_encoder_lr> <selected_weight_decay>"
  echo "Set RFDETR_PLAN_ONLY=1 to write and inspect a plan without training."
  exit 2
fi

REPO_DIR="${RFDETR_REPO_DIR:-/work/projects/myproj}"
ENV_FILE="${REPO_DIR}/rfdetr_hpo_env_ucloud.sh"
if [ -f "${ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

PROJECT_DIR="${PROJECT_DIR:-${REPO_DIR}/SOLO_Supervised_RFDETR}"
PYTHON_BIN="${RFDETR_PYTHON:-/work/CondaEnv/envs/rfdetr180/bin/python}"
TRAIN_SCRIPT="${PROJECT_DIR}/TrainRFDETR_SOLO_SingleClass_STATIC_Ucloud.py"

detect_user_base() {
  if compgen -G "/work/Member Files:*" >/dev/null; then
    basename "$(ls -d /work/Member\ Files:* | head -n1)"
  elif compgen -G "/work/*#*" >/dev/null; then
    basename "$(ls -d /work/*#* | head -n1)"
  else
    echo ""
  fi
}

USER_BASE_DIR="${USER_BASE_DIR:-$(detect_user_base)}"
if [ -z "${USER_BASE_DIR}" ]; then
  echo "[RFDETR run][ERROR] Could not locate the UCloud member-files directory under /work." >&2
  exit 1
fi

export RFDETR_RUNTIME_PROFILE="ucloud"
export RFDETR_EXPERIMENT_MODE="focused_hpo"
export RFDETR_FOCUSED_STAGE="${STAGE}"
export RFDETR_HPO_TARGET="two-class"
export RFDETR_MODEL_CLS="RFDETRLarge"
export RFDETR_INPUT_MODE="640"
export RFDETR_PREFER_MODEL_NATIVE_DEFAULTS="1"
export RFDETR_ALLOW_NON_NATIVE_PRETRAIN_RESOLUTION="0"
export RFDETR_RUN_TEST="0"
export RFDETR_FOCUSED_SEED="42"
export RFDETR_FOCUSED_EPOCHS="120"
export RFDETR_FOCUSED_BATCH="${RFDETR_FOCUSED_BATCH:-4}"
export RFDETR_FOCUSED_GRAD_ACCUM_STEPS="${RFDETR_FOCUSED_GRAD_ACCUM_STEPS:-4}"
export RFDETR_FOCUSED_WARMUP_EPOCHS="3"
export RFDETR_FOCUSED_LR_SCHEDULER="cosine"
export RFDETR_FOCUSED_PATIENCE="20"
export RFDETR_FOCUSED_MIN_DELTA="0.001"
export RFDETR_OOM_MAX_RETRIES="0"
export MAX_PARALLEL="1"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export DATASET_TWO_CLASS="${DATASET_TWO_CLASS:-${PROJECT_DIR}/Stat_Dataset/QA-2025v1_TwoClass_OVR_V2_20260618-101346}"
export IMAGES_FALLBACK_ROOT="${IMAGES_FALLBACK_ROOT:-/work/${USER_BASE_DIR}/CellScanData/Zoom10x - Quality Assessment_Cleaned}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/work/${USER_BASE_DIR}/RFDETR_SOLO_OUTPUT/FOCUSED_LARGE_HPO/${STAGE^^}}"

if [ "${STAGE}" = "wd" ] || [ "${STAGE}" = "final" ]; then
  if [ -z "${SELECTED_LR}" ] || [ -z "${SELECTED_ENCODER_LR}" ]; then
    echo "[RFDETR run][ERROR] ${STAGE} requires selected main and encoder learning rates." >&2
    exit 2
  fi
  export RFDETR_FOCUSED_SELECTED_LR="${SELECTED_LR}"
  export RFDETR_FOCUSED_SELECTED_LR_ENCODER="${SELECTED_ENCODER_LR}"
fi

if [ "${STAGE}" = "final" ]; then
  if [ -z "${SELECTED_WEIGHT_DECAY}" ]; then
    echo "[RFDETR run][ERROR] final requires the selected weight decay." >&2
    exit 2
  fi
  export RFDETR_FOCUSED_SELECTED_WEIGHT_DECAY="${SELECTED_WEIGHT_DECAY}"
  export RFDETR_RUN_TEST="1"
fi

for required in "${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${DATASET_TWO_CLASS}" "${IMAGES_FALLBACK_ROOT}"; do
  if [ ! -e "${required}" ]; then
    echo "[RFDETR run][ERROR] Required path does not exist: ${required}" >&2
    exit 1
  fi
done

echo "[RFDETR run] Stage: ${STAGE}"
echo "[RFDETR run] GPU concurrency: ${MAX_PARALLEL}"
echo "[RFDETR run] Dataset: ${DATASET_TWO_CLASS}"
echo "[RFDETR run] Source images: ${IMAGES_FALLBACK_ROOT}"
echo "[RFDETR run] Output: ${OUTPUT_ROOT}"
echo "[RFDETR run] Plan only: ${RFDETR_PLAN_ONLY:-0}"

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" -u "${TRAIN_SCRIPT}"
