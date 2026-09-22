#!/usr/bin/env bash
set -Eeuo pipefail

# Arm 2: official public DINOv3-S/16 backbone + randomly initialized RF-DETR Small.
PROJECT_DIR="${PROJECT_DIR:-/work/projects/myproj}"
ENV_FILE="$PROJECT_DIR/dinov3_ucloud_env.sh"
if [[ -f "$ENV_FILE" ]]; then source "$ENV_FILE"; fi

STUDY3_DIR="${STUDY3_DIR:-$PROJECT_DIR/Study3_SelfSupervised}"
DINOV3_REPO="${DINOV3_REPO:-/work/projects/dinov3}"
IMAGE_ROOT="${IMAGE_ROOT:-/work/40x Input tiles for CVAT}"
DATASET_DIR="${STUDY3_DETECTION_DATASET:-$PROJECT_DIR/SOLO_Supervised_RFDETR/Stat_Dataset40x/QA_40x-_20260901-135953}"
CONFIG_PATH="${STUDY3_PUBLIC_SSL_CONFIG:-$STUDY3_DIR/detection_public_ssl_ucloud_config.json}"
RUN_OUTPUT_ROOT="${STUDY3_DETECTION_OUTPUT:-${OUTPUT_ROOT:-/work/DINOv3_Study3_OUTPUT}/DetectionRFDETR}"
WEIGHTS_FILENAME="dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

resolve_public_weights() {
  if [[ -n "${STUDY3_PUBLIC_SSL_WEIGHTS:-}" ]]; then
    printf '%s\n' "$STUDY3_PUBLIC_SSL_WEIGHTS"
    return 0
  fi
  shopt -s nullglob
  local candidates=(
    "${OUTPUT_ROOT:-/work/DINOv3_Study3_OUTPUT}/public_weights/$WEIGHTS_FILENAME"
    /work/Member\ Files:*/Checkpoints/Pretrained_Models/"$WEIGHTS_FILENAME"
    /work/Checkpoints/Pretrained_Models/"$WEIGHTS_FILENAME"
    /work/Pretrained_Models/"$WEIGHTS_FILENAME"
  )
  shopt -u nullglob
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if ! PUBLIC_WEIGHTS="$(resolve_public_weights)"; then
  echo "[Study3 public SSL][ERROR] Could not locate $WEIGHTS_FILENAME." >&2
  echo "Attach the Pretrained_Models folder when starting the job, or set STUDY3_PUBLIC_SSL_WEIGHTS." >&2
  exit 1
fi

# Persist this download. The first run downloads the released official backbone.
export TORCH_HOME="${DINOV3_PUBLIC_WEIGHTS_CACHE:-${OUTPUT_ROOT:-/work/DINOv3_Study3_OUTPUT}/torch_hub}"
for required in "$DINOV3_REPO/hubconf.py" "$DATASET_DIR/train/_annotations.coco.json" "$DATASET_DIR/valid/_annotations.coco.json" "$IMAGE_ROOT" "$CONFIG_PATH" "$PUBLIC_WEIGHTS"; do
  if [[ ! -e "$required" ]]; then
    echo "[Study3 public SSL][ERROR] Required path not found: $required" >&2
    exit 1
  fi
done

mkdir -p "$RUN_OUTPUT_ROOT" "$TORCH_HOME"
echo "[Study3 public SSL] Python: $(command -v python)"
echo "[Study3 public SSL] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
echo "[Study3 public SSL] Config: $CONFIG_PATH"
echo "[Study3 public SSL] Output: $RUN_OUTPUT_ROOT"
echo "[Study3 public SSL] Official weights cache: $TORCH_HOME"
echo "[Study3 public SSL] Official weights: $PUBLIC_WEIGHTS"

exec python "$STUDY3_DIR/run_detection_experiments.py" \
  --config "$CONFIG_PATH" --arms public_ssl \
  --dinov3-repo "$DINOV3_REPO" --dataset-dir "$DATASET_DIR" \
  --image-root "$IMAGE_ROOT" --output-root "$RUN_OUTPUT_ROOT" \
  --public-ssl-weights "$PUBLIC_WEIGHTS" --train
