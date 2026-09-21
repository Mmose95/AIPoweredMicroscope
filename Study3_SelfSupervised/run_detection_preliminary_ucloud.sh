#!/usr/bin/env bash
set -Eeuo pipefail

# Run the 40x preliminary comparison on one UCloud GPU. Both arms use the
# fixed training/validation split; the test set is excluded by the Python
# launcher and is never evaluated here.

PROJECT_DIR="${PROJECT_DIR:-/work/projects/myproj}"
ENV_FILE="$PROJECT_DIR/dinov3_ucloud_env.sh"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

STUDY3_DIR="${STUDY3_DIR:-$PROJECT_DIR/Study3_SelfSupervised}"
DINOV3_REPO="${DINOV3_REPO:-/work/projects/dinov3}"
IMAGE_ROOT="${IMAGE_ROOT:-/work/40x Input tiles for CVAT}"
DATASET_DIR="${STUDY3_DETECTION_DATASET:-$PROJECT_DIR/SOLO_Supervised_RFDETR/Stat_Dataset40x/QA_40x-_20260901-135953}"
CONFIG_PATH="${STUDY3_DETECTION_CONFIG:-$STUDY3_DIR/detection_preliminary_ucloud_config.json}"
RUN_OUTPUT_ROOT="${STUDY3_DETECTION_OUTPUT:-${OUTPUT_ROOT:-/work/DINOv3_Study3_OUTPUT}/DetectionRFDETR}"
SSL_CHECKPOINT="${STUDY3_SSL_CHECKPOINT:-${OUTPUT_ROOT:-/work/DINOv3_Study3_OUTPUT}/dinov3_vits16_full_seed0/eval/training_81623/teacher_checkpoint.pth}"

for required in \
  "$DINOV3_REPO/hubconf.py" \
  "$DATASET_DIR/train/_annotations.coco.json" \
  "$DATASET_DIR/valid/_annotations.coco.json" \
  "$IMAGE_ROOT" \
  "$CONFIG_PATH" \
  "$SSL_CHECKPOINT"
do
  if [[ ! -e "$required" ]]; then
    echo "[Study3 preliminary][ERROR] Required path not found: $required" >&2
    echo "Set STUDY3_SSL_CHECKPOINT or STUDY3_DETECTION_OUTPUT if the teacher checkpoint is stored elsewhere." >&2
    exit 1
  fi
done

mkdir -p "$RUN_OUTPUT_ROOT"
echo "[Study3 preliminary] Python: $(command -v python)"
echo "[Study3 preliminary] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "[Study3 preliminary] Config: $CONFIG_PATH"
echo "[Study3 preliminary] Output: $RUN_OUTPUT_ROOT"
echo "[Study3 preliminary] Teacher: $SSL_CHECKPOINT"

exec python "$STUDY3_DIR/run_detection_experiments.py" \
  --config "$CONFIG_PATH" \
  --dinov3-repo "$DINOV3_REPO" \
  --dataset-dir "$DATASET_DIR" \
  --image-root "$IMAGE_ROOT" \
  --ssl-checkpoint "$SSL_CHECKPOINT" \
  --output-root "$RUN_OUTPUT_ROOT" \
  --train
