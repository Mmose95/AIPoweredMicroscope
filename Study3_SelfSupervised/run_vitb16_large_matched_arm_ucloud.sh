#!/usr/bin/env bash
# Train one matched DINOv3-B/16 + RF-DETR Large comparison arm on UCloud.
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

ARM="${1:-${STUDY3_DETECTION_ARM:-}}"
case "$ARM" in
  scratch|public_ssl|own_data_ssl) ;;
  *)
    echo "Usage: bash $0 {scratch|public_ssl|own_data_ssl}" >&2
    exit 2
    ;;
esac

DATASET_DIR="${STUDY3_DETECTION_DATASET:-$PROJECT_DIR/SOLO_Supervised_RFDETR/Stat_Dataset40x/QA_40x-_20260901-135953}"
CONFIG_PATH="${STUDY3_DETECTION_CONFIG:-$STUDY3_DIR/detection_vitb16_large_matched_arms_ucloud_config.json}"
SSL_CHECKPOINT="${STUDY3_SSL_CHECKPOINT:-$OUTPUT_ROOT/dinov3_vitb16_full_seed0/eval/training_161099/teacher_checkpoint.pth}"
RUN_OUTPUT_ROOT="${STUDY3_DETECTION_OUTPUT:-$OUTPUT_ROOT/DetectionRFDETR}"
EXPECTED_SSL_SHA256="a5ce211ed25c73ad4929848d8ab93751d10eae9ffb66ada1c61ba49a764b14f0"

find_public_weights() {
  local candidate
  if [[ -n "${STUDY3_PUBLIC_SSL_WEIGHTS:-}" ]]; then
    printf '%s\n' "$STUDY3_PUBLIC_SSL_WEIGHTS"
    return
  fi
  while IFS= read -r candidate; do
    [[ -n "$candidate" ]] && { printf '%s\n' "$candidate"; return; }
  done < <(find /work -maxdepth 4 -type f -name 'dinov3_vitb16_pretrain*.pth' -print 2>/dev/null | sort)
}

PUBLIC_WEIGHTS=""
if [[ "$ARM" == "public_ssl" ]]; then
  PUBLIC_WEIGHTS="$(find_public_weights)"
  [[ -n "$PUBLIC_WEIGHTS" && -f "$PUBLIC_WEIGHTS" ]] || {
    echo "[Matched arm][ERROR] Official DINOv3-B/16 weights were not found." >&2
    echo "Attach the Pretrained_Models folder or set STUDY3_PUBLIC_SSL_WEIGHTS." >&2
    echo "The required file name begins with dinov3_vitb16_pretrain (not dinov3_vits16)." >&2
    exit 1
  }
fi

for required in \
  "$DINOV3_REPO/hubconf.py" \
  "$DATASET_DIR/train/_annotations.coco.json" \
  "$DATASET_DIR/valid/_annotations.coco.json" \
  "$IMAGE_ROOT" \
  "$CONFIG_PATH"
do
  [[ -e "$required" ]] || { echo "[Matched arm][ERROR] Missing: $required" >&2; exit 1; }
done

EXTRA_ARGS=()
if [[ "$ARM" == "public_ssl" ]]; then
  EXTRA_ARGS+=(--public-ssl-weights "$PUBLIC_WEIGHTS")
elif [[ "$ARM" == "own_data_ssl" ]]; then
  [[ -f "$SSL_CHECKPOINT" ]] || { echo "[Matched arm][ERROR] Missing: $SSL_CHECKPOINT" >&2; exit 1; }
  actual_sha256="$(sha256sum "$SSL_CHECKPOINT" | awk '{print $1}')"
  [[ "$actual_sha256" == "$EXPECTED_SSL_SHA256" ]] || {
    echo "[Matched arm][ERROR] Own-data checkpoint is not the selected epoch-150 checkpoint." >&2
    echo "Expected: $EXPECTED_SSL_SHA256" >&2
    echo "Actual:   $actual_sha256" >&2
    exit 1
  }
  EXTRA_ARGS+=(--ssl-checkpoint "$SSL_CHECKPOINT")
fi

mkdir -p "$RUN_OUTPUT_ROOT"
cat <<EOF
[Matched arm] Arm: $ARM
[Matched arm] Python: $(command -v python)
[Matched arm] Visible GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')
[Matched arm] Config: $CONFIG_PATH
[Matched arm] Dataset: $DATASET_DIR
[Matched arm] Images: $IMAGE_ROOT
[Matched arm] Public weights: ${PUBLIC_WEIGHTS:-not used}
[Matched arm] Own-data checkpoint: $([[ "$ARM" == "own_data_ssl" ]] && printf '%s' "$SSL_CHECKPOINT" || printf 'not used')
[Matched arm] Output root: $RUN_OUTPUT_ROOT
[Matched arm] Test split: unused
EOF

exec python "$STUDY3_DIR/run_detection_experiments.py" \
  --config "$CONFIG_PATH" \
  --arms "$ARM" \
  --dinov3-repo "$DINOV3_REPO" \
  --dataset-dir "$DATASET_DIR" \
  --image-root "$IMAGE_ROOT" \
  --output-root "$RUN_OUTPUT_ROOT" \
  "${EXTRA_ARGS[@]}" \
  --train
