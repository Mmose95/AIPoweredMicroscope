#!/usr/bin/env bash
# Fully fine-tune RF-DETR Large and DINOv3-B/16 from the selected SSL epoch 150.
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/work/projects/myproj}"
ENV_FILE="$PROJECT_DIR/dinov3_ucloud_env.sh"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

: "${DINOV3_REPO:?DINOV3_REPO is unset; source dinov3_ucloud_env.sh}"
: "${STUDY3_DIR:?STUDY3_DIR is unset; source dinov3_ucloud_env.sh}"
: "${IMAGE_ROOT:?IMAGE_ROOT is unset; source dinov3_ucloud_env.sh}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is unset; source dinov3_ucloud_env.sh}"

DATASET_DIR="${STUDY3_DETECTION_DATASET:-$PROJECT_DIR/SOLO_Supervised_RFDETR/Stat_Dataset40x/QA_40x-_20260901-135953}"
CONFIG_PATH="${STUDY3_DETECTION_CONFIG:-$STUDY3_DIR/detection_vitb16_large_epoch150_full_ucloud_config.json}"
SSL_CHECKPOINT="${STUDY3_SSL_CHECKPOINT:-$OUTPUT_ROOT/dinov3_vitb16_full_seed0/eval/training_161099/teacher_checkpoint.pth}"
RUN_OUTPUT_ROOT="${STUDY3_DETECTION_OUTPUT:-$OUTPUT_ROOT/DetectionRFDETR}"
EXPECTED_SHA256="a5ce211ed25c73ad4929848d8ab93751d10eae9ffb66ada1c61ba49a764b14f0"

for required in \
  "$DINOV3_REPO/hubconf.py" \
  "$DATASET_DIR/train/_annotations.coco.json" \
  "$DATASET_DIR/valid/_annotations.coco.json" \
  "$IMAGE_ROOT" \
  "$CONFIG_PATH" \
  "$SSL_CHECKPOINT"
do
  [[ -e "$required" ]] || {
    echo "[Full fine-tune][ERROR] Required path not found: $required" >&2
    exit 1
  }
done

actual_sha256="$(sha256sum "$SSL_CHECKPOINT" | awk '{print $1}')"
[[ "$actual_sha256" == "$EXPECTED_SHA256" ]] || {
  echo "[Full fine-tune][ERROR] Selected SSL checkpoint hash does not match epoch 150." >&2
  echo "Expected: $EXPECTED_SHA256" >&2
  echo "Actual:   $actual_sha256" >&2
  exit 1
}

mkdir -p "$RUN_OUTPUT_ROOT"
cat <<EOF
[Full fine-tune] Python: $(command -v python)
[Full fine-tune] GPU count visible: $(python -c 'import torch; print(torch.cuda.device_count())')
[Full fine-tune] Config: $CONFIG_PATH
[Full fine-tune] Dataset: $DATASET_DIR
[Full fine-tune] Images: $IMAGE_ROOT
[Full fine-tune] SSL checkpoint: $SSL_CHECKPOINT
[Full fine-tune] SSL checkpoint SHA-256: $actual_sha256
[Full fine-tune] Output root: $RUN_OUTPUT_ROOT
[Full fine-tune] Test split: unused
EOF

exec python "$STUDY3_DIR/run_detection_experiments.py" \
  --config "$CONFIG_PATH" \
  --arms own_data_ssl \
  --dinov3-repo "$DINOV3_REPO" \
  --dataset-dir "$DATASET_DIR" \
  --image-root "$IMAGE_ROOT" \
  --ssl-checkpoint "$SSL_CHECKPOINT" \
  --output-root "$RUN_OUTPUT_ROOT" \
  --train
