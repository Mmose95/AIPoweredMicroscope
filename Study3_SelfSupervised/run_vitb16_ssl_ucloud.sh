#!/usr/bin/env bash
# Generate and launch the full Study 3 DINOv3-B/16 SSL run on UCloud.
set -Eeuo pipefail

: "${DINOV3_REPO:?Source dinov3_ucloud_env.sh first (DINOV3_REPO is unset)}"
: "${STUDY3_DIR:?Source dinov3_ucloud_env.sh first (STUDY3_DIR is unset)}"
: "${FULL_MANIFEST:?Source dinov3_ucloud_env.sh first (FULL_MANIFEST is unset)}"
: "${IMAGE_ROOT:?Source dinov3_ucloud_env.sh first (IMAGE_ROOT is unset)}"
: "${OUTPUT_ROOT:?Source dinov3_ucloud_env.sh first (OUTPUT_ROOT is unset)}"

gpu_count="${STUDY3_SSL_GPUS:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
batch_per_gpu="${STUDY3_SSL_BATCH_PER_GPU:-32}"
epochs="${STUDY3_SSL_EPOCHS:-100}"
warmup_epochs="${STUDY3_SSL_WARMUP_EPOCHS:-10}"
num_workers="${STUDY3_SSL_NUM_WORKERS:-8}"
seed="${STUDY3_SSL_SEED:-0}"
resume="${STUDY3_SSL_RESUME:-0}"

[[ "$gpu_count" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid GPU count: $gpu_count" >&2; exit 2; }
[[ "$batch_per_gpu" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid per-GPU batch: $batch_per_gpu" >&2; exit 2; }

if [[ "${STUDY3_SSL_ALLOW_MIG:-0}" != "1" ]] && \
   python -c 'import torch,sys; sys.exit(0 if any("MIG" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) else 1)'
then
  echo "[Study3 SSL][ERROR] A MIG partition was detected." >&2
  echo "Use full B200 GPUs for the definitive B/16 run, or set STUDY3_SSL_ALLOW_MIG=1 for a smoke test." >&2
  exit 2
fi

config_dir="$OUTPUT_ROOT/generated_configs"
config_path="$config_dir/dinov3_vitb16_${gpu_count}gpu_b${batch_per_gpu}_e${epochs}_seed${seed}.yaml"
run_dir="${STUDY3_VITB16_SSL_OUTPUT:-$OUTPUT_ROOT/dinov3_vitb16_full_seed${seed}}"
mkdir -p "$config_dir"

if [[ -d "$run_dir" ]] && [[ -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]] && [[ "$resume" != "1" ]]; then
  echo "[Study3 SSL][ERROR] Output directory is not empty: $run_dir" >&2
  echo "Choose another STUDY3_VITB16_SSL_OUTPUT or set STUDY3_SSL_RESUME=1 to resume." >&2
  exit 2
fi

python "$STUDY3_DIR/make_ucloud_training_config.py" \
  --base-config "$STUDY3_DIR/configs/dinov3_vitb16_ucloud_base.yaml" \
  --manifest "$FULL_MANIFEST" \
  --output "$config_path" \
  --gpus "$gpu_count" \
  --batch-size-per-gpu "$batch_per_gpu" \
  --epochs "$epochs" \
  --warmup-epochs "$warmup_epochs" \
  --num-workers "$num_workers" \
  --seed "$seed"

echo "[Study3 SSL] Architecture: DINOv3-B/16"
echo "[Study3 SSL] GPUs: $gpu_count"
echo "[Study3 SSL] Per-GPU batch: $batch_per_gpu"
echo "[Study3 SSL] Global batch: $((gpu_count * batch_per_gpu))"
echo "[Study3 SSL] Seed: $seed"
echo "[Study3 SSL] Output: $run_dir"
echo "[Study3 SSL] Resume: $resume"

export DINOV3_RESUME="$resume"
bash "$STUDY3_DIR/run_official_dinov3_linux.sh" \
  "$DINOV3_REPO" \
  "$config_path" \
  "$FULL_MANIFEST" \
  "$IMAGE_ROOT" \
  "$run_dir" \
  "$gpu_count"
