#!/usr/bin/env bash
# Continue the completed 100-epoch Study 3 DINOv3-B/16 run to epoch 200.
#
# This is a low-learning-rate continuation phase. It resumes the complete
# official checkpoint (student, EMA teacher, SSL heads, and optimizer) in the
# original run directory. It intentionally does not reuse the original peak
# learning rate, because changing a finished 100-epoch cosine schedule to 200
# epochs would otherwise produce a large learning-rate jump at resume.
set -Eeuo pipefail

: "${DINOV3_REPO:?Source dinov3_ucloud_env.sh first (DINOV3_REPO is unset)}"
: "${STUDY3_DIR:?Source dinov3_ucloud_env.sh first (STUDY3_DIR is unset)}"
: "${FULL_MANIFEST:?Source dinov3_ucloud_env.sh first (FULL_MANIFEST is unset)}"
: "${IMAGE_ROOT:?Source dinov3_ucloud_env.sh first (IMAGE_ROOT is unset)}"
: "${OUTPUT_ROOT:?Source dinov3_ucloud_env.sh first (OUTPUT_ROOT is unset)}"

# Match the completed run exactly. A different global batch changes the
# iteration-to-epoch mapping and would invalidate checkpoint comparisons.
gpu_count="${STUDY3_SSL_GPUS:-2}"
batch_per_gpu="${STUDY3_SSL_BATCH_PER_GPU:-64}"
seed="${STUDY3_SSL_SEED:-0}"
num_workers="${STUDY3_SSL_NUM_WORKERS:-8}"
target_epoch="${STUDY3_SSL_TARGET_EPOCH:-200}"
original_final_iteration="${STUDY3_SSL_ORIGINAL_FINAL_ITERATION:-107399}"
run_dir="${STUDY3_VITB16_SSL_OUTPUT:-$OUTPUT_ROOT/dinov3_vitb16_full_seed${seed}}"

# Raw LR before DINOv3's sqrt(global_batch/1024) scaling. With global batch
# 128, 1e-4 becomes 1.414e-4 at the top of the reconstructed 200-epoch
# schedule and about 7.1e-5 at epoch 100, where continuation begins.
continuation_lr="${STUDY3_SSL_CONTINUATION_LR:-0.0001}"
continuation_min_lr="${STUDY3_SSL_CONTINUATION_MIN_LR:-0.000001}"

[[ "$gpu_count" == "2" ]] || {
  echo "[Continuation][ERROR] Expected 2 GPUs to match the original run; received $gpu_count." >&2
  exit 2
}
[[ "$batch_per_gpu" == "64" ]] || {
  echo "[Continuation][ERROR] Expected batch size 64 per GPU; received $batch_per_gpu." >&2
  exit 2
}
[[ "$seed" == "0" ]] || {
  echo "[Continuation][ERROR] Expected seed 0; received $seed." >&2
  exit 2
}
[[ "$target_epoch" =~ ^[0-9]+$ ]] && (( target_epoch > 100 )) || {
  echo "[Continuation][ERROR] STUDY3_SSL_TARGET_EPOCH must be an integer greater than 100." >&2
  exit 2
}

visible_gpus="$(python -c 'import torch; print(torch.cuda.device_count())')"
(( visible_gpus >= gpu_count )) || {
  echo "[Continuation][ERROR] PyTorch sees $visible_gpus GPU(s), but $gpu_count are required." >&2
  exit 2
}

latest_ckpt="$(find "$run_dir/ckpt" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort -n | tail -n 1)"
[[ "$latest_ckpt" =~ ^[0-9]+$ ]] || {
  echo "[Continuation][ERROR] No numeric checkpoint was found in $run_dir/ckpt." >&2
  exit 2
}
(( latest_ckpt >= original_final_iteration )) || {
  echo "[Continuation][ERROR] Expected checkpoint $original_final_iteration or later, found $latest_ckpt." >&2
  echo "[Continuation][ERROR] Run directory: $run_dir" >&2
  exit 2
}
target_final_iteration=$((target_epoch * 1074 - 1))
(( latest_ckpt < target_final_iteration )) || {
  echo "[Continuation] Target epoch $target_epoch is already complete (checkpoint $latest_ckpt)."
  exit 0
}

config_dir="$OUTPUT_ROOT/generated_configs"
config_path="$config_dir/dinov3_vitb16_2gpu_b64_e${target_epoch}_seed0_continuation.yaml"
mkdir -p "$config_dir"

python "$STUDY3_DIR/make_ucloud_training_config.py" \
  --base-config "$STUDY3_DIR/configs/dinov3_vitb16_ucloud_base.yaml" \
  --manifest "$FULL_MANIFEST" \
  --output "$config_path" \
  --gpus "$gpu_count" \
  --batch-size-per-gpu "$batch_per_gpu" \
  --epochs "$target_epoch" \
  --warmup-epochs 0 \
  --num-workers "$num_workers" \
  --seed "$seed"

cat <<EOF
[Continuation] Resume checkpoint: $run_dir/ckpt/$latest_ckpt
[Continuation] Target epoch: $target_epoch
[Continuation] GPUs: $gpu_count
[Continuation] Per-GPU batch: $batch_per_gpu
[Continuation] Global batch: $((gpu_count * batch_per_gpu))
[Continuation] Raw continuation LR: $continuation_lr
[Continuation] Minimum LR: $continuation_min_lr
[Continuation] Output: $run_dir
EOF

export DINOV3_RESUME=1
bash "$STUDY3_DIR/run_official_dinov3_linux.sh" \
  "$DINOV3_REPO" \
  "$config_path" \
  "$FULL_MANIFEST" \
  "$IMAGE_ROOT" \
  "$run_dir" \
  "$gpu_count" \
  "optim.lr=$continuation_lr" \
  "optim.min_lr=$continuation_min_lr" \
  "optim.weight_decay=0.4" \
  "optim.weight_decay_end=0.4" \
  "teacher.momentum_teacher=0.992" \
  "teacher.final_momentum_teacher=1.0"
