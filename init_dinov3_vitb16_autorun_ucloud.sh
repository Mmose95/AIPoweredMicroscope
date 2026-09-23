#!/usr/bin/env bash
# UCloud initialization file for an unattended queued DINOv3-B/16 SSL job.
# It prepares the normal Study 3 environment, verifies the requested full-GPU
# allocation, then launches training under nohup so Jupyter can finish starting.
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/projects}"
PROJECT_DIR="${PROJECT_DIR:-$PROJECT_ROOT/myproj}"
PROJECT_REPO_URL="${PROJECT_REPO_URL:-https://github.com/Mmose95/AIPoweredMicroscope}"
EXPECTED_GPUS="${STUDY3_SSL_EXPECTED_GPUS:-4}"

mkdir -p "$PROJECT_ROOT"
if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  echo "[DINOv3 B/16 autorun] Cloning project"
  git clone "$PROJECT_REPO_URL" "$PROJECT_DIR"
else
  echo "[DINOv3 B/16 autorun] Updating project"
  git -C "$PROJECT_DIR" pull --ff-only
fi

# Run the ordinary initializer first. It creates/activates the environment,
# validates CUDA, locates the mounted data, and writes dinov3_ucloud_env.sh.
bash "$PROJECT_DIR/init_dinov3_ucloud.sh"
source "$PROJECT_DIR/dinov3_ucloud_env.sh"

actual_gpus="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$actual_gpus" != "$EXPECTED_GPUS" ]]; then
  echo "[DINOv3 B/16 autorun][ERROR] Expected $EXPECTED_GPUS GPUs, found $actual_gpus." >&2
  exit 2
fi
if python -c 'import torch,sys; sys.exit(0 if any("MIG" in torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())) else 1)'
then
  echo "[DINOv3 B/16 autorun][ERROR] MIG devices were allocated; full GPUs are required." >&2
  exit 2
fi

export STUDY3_SSL_GPUS="$EXPECTED_GPUS"
export STUDY3_SSL_BATCH_PER_GPU="${STUDY3_SSL_BATCH_PER_GPU:-32}"
export STUDY3_SSL_EPOCHS="${STUDY3_SSL_EPOCHS:-100}"
export STUDY3_SSL_WARMUP_EPOCHS="${STUDY3_SSL_WARMUP_EPOCHS:-10}"
export STUDY3_SSL_NUM_WORKERS="${STUDY3_SSL_NUM_WORKERS:-8}"
export STUDY3_SSL_SEED="${STUDY3_SSL_SEED:-0}"
export STUDY3_SSL_RESUME="${STUDY3_SSL_RESUME:-0}"
export PYTHONUNBUFFERED=1

log_dir="$OUTPUT_ROOT/logs"
mkdir -p "$log_dir"
log_path="$log_dir/dinov3_vitb16_full_seed${STUDY3_SSL_SEED}.log"
pid_path="$log_dir/dinov3_vitb16_full_seed${STUDY3_SSL_SEED}.pid"

echo "[DINOv3 B/16 autorun] Launching unattended training"
echo "[DINOv3 B/16 autorun] Log: $log_path"
nohup bash "$STUDY3_DIR/run_vitb16_ssl_ucloud.sh" >"$log_path" 2>&1 &
training_pid=$!
printf '%s\n' "$training_pid" >"$pid_path"

# Detect immediate configuration/launch failures while keeping the long run
# detached from the initialization process.
sleep 5
if ! kill -0 "$training_pid" 2>/dev/null; then
  echo "[DINOv3 B/16 autorun][ERROR] Training exited during startup." >&2
  tail -100 "$log_path" >&2 || true
  exit 1
fi

echo "[DINOv3 B/16 autorun] Training PID: $training_pid"
echo "[DINOv3 B/16 autorun] Monitor with: tail -f '$log_path'"
echo "[DINOv3 B/16 autorun] Initialization complete; training continues in the background."
