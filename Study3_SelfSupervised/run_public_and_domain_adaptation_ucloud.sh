#!/usr/bin/env bash
set -Eeuo pipefail

# Runs only the pending arms: (2) public backbone and (4) public -> domain SSL.
# Existing scratch and own-data SSL detector outputs are never touched.
PROJECT_DIR="${PROJECT_DIR:-/work/projects/myproj}"
ENV_FILE="$PROJECT_DIR/dinov3_ucloud_env.sh"
if [[ -f "$ENV_FILE" ]]; then source "$ENV_FILE"; fi
STUDY3_DIR="${STUDY3_DIR:-$PROJECT_DIR/Study3_SelfSupervised}"
DINOV3_REPO="${DINOV3_REPO:-/work/projects/dinov3}"
IMAGE_ROOT="${IMAGE_ROOT:-/work/40x Input tiles for CVAT}"
MANIFEST="${FULL_MANIFEST:-$STUDY3_DIR/manifests/ssl_pool_40x_9d8cb0d9ec7b.csv}"
DATASET_DIR="${STUDY3_DETECTION_DATASET:-$PROJECT_DIR/SOLO_Supervised_RFDETR/Stat_Dataset40x/QA_40x-_20260901-135953}"
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
  echo "[Study3 arms 2+4][ERROR] Could not locate $WEIGHTS_FILENAME." >&2
  echo "Attach the Pretrained_Models folder when starting the job, or set STUDY3_PUBLIC_SSL_WEIGHTS." >&2
  exit 1
fi
ADAPTATION_OUTPUT="${STUDY3_PUBLIC_ADAPTATION_OUTPUT:-${OUTPUT_ROOT:-/work/DINOv3_Study3_OUTPUT}/dinov3_vits16_public_domain_adaptation_seed0}"
ADAPTATION_CONFIG="${STUDY3_PUBLIC_ADAPTATION_CONFIG:-$STUDY3_DIR/configs/dinov3_vits16_public_domain_adaptation_1gpu.yaml}"
GPUS="${STUDY3_PUBLIC_ADAPTATION_GPUS:-1}"

for required in "$DINOV3_REPO/hubconf.py" "$IMAGE_ROOT" "$MANIFEST" "$PUBLIC_WEIGHTS" "$DATASET_DIR/train/_annotations.coco.json" "$DATASET_DIR/valid/_annotations.coco.json" "$ADAPTATION_CONFIG"; do
  [[ -e "$required" ]] || { echo "[Study3 arms 2+4][ERROR] Missing: $required" >&2; exit 1; }
done

PUBLIC_RUN_GLOB="$RUN_OUTPUT_ROOT/study3_dinov3_rfdetr_public_ssl_40x"/*/public_ssl/seed_0/budget_1p000/run_record.json
if grep -q '"status": "completed"' $PUBLIC_RUN_GLOB 2>/dev/null; then
  echo "[Study3 arms 2+4] Arm 2 already completed; retaining its existing output."
else
  echo "[Study3 arms 2+4] First: arm 2 public DINOv3 + random RF-DETR Small"
  bash "$STUDY3_DIR/run_detection_public_ssl_ucloud.sh"
fi

if TEACHER_CHECKPOINT="$(python "$STUDY3_DIR/find_latest_teacher_checkpoint.py" --run-dir "$ADAPTATION_OUTPUT" 2>/dev/null)"; then
  echo "[Study3 arms 2+4] Arm 4 SSL teacher already exists; retaining it."
else
  echo "[Study3 arms 2+4] Second: arm 4 public DINOv3 -> microscopy SSL adaptation"
  bash "$STUDY3_DIR/run_official_dinov3_linux.sh" \
    "$DINOV3_REPO" "$ADAPTATION_CONFIG" "$MANIFEST" "$IMAGE_ROOT" \
    "$ADAPTATION_OUTPUT" "$GPUS" \
    --public-backbone-weights "$PUBLIC_WEIGHTS"
  TEACHER_CHECKPOINT="$(python "$STUDY3_DIR/find_latest_teacher_checkpoint.py" --run-dir "$ADAPTATION_OUTPUT")"
fi
echo "[Study3 arms 2+4] Arm 4 teacher: $TEACHER_CHECKPOINT"
ADAPTED_RUN_GLOB="$RUN_OUTPUT_ROOT/study3_dinov3_rfdetr_public_domain_ssl_40x"/*/public_domain_ssl/seed_0/budget_1p000/run_record.json
if grep -q '"status": "completed"' $ADAPTED_RUN_GLOB 2>/dev/null; then
  echo "[Study3 arms 2+4] Arm 4 detector already completed; retaining its existing output."
else
  python "$STUDY3_DIR/run_detection_experiments.py" \
    --config "$STUDY3_DIR/detection_public_domain_ssl_ucloud_config.json" \
    --arms public_domain_ssl --dinov3-repo "$DINOV3_REPO" \
    --dataset-dir "$DATASET_DIR" --image-root "$IMAGE_ROOT" \
    --ssl-checkpoint "$TEACHER_CHECKPOINT" --public-ssl-weights "$PUBLIC_WEIGHTS" \
    --output-root "$RUN_OUTPUT_ROOT" --train
fi
