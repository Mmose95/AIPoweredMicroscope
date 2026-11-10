#!/usr/bin/env bash
# Shared preprocessing and individual jobs for the corrected 40x experiment.
set -Eeuo pipefail

study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$study_dir")"
if [[ -f "$repo_dir/dinov3_ucloud_env.sh" ]]; then
  source "$repo_dir/dinov3_ucloud_env.sh"
fi

mode="${1:-}"
case "$mode" in
  prepare|scratch|public_ssl|ssl|ssl-scratch|ssl-adapt|own_data_ssl|public_domain_ssl) ;;
  *) echo "Usage: $0 {prepare|scratch|public_ssl|ssl|ssl-scratch|ssl-adapt|own_data_ssl|public_domain_ssl}" >&2; exit 2 ;;
esac

: "${OUTPUT_ROOT:?Source the UCloud environment or set OUTPUT_ROOT to persistent storage}"
work="$OUTPUT_ROOT/Native704_20261110"
source_split="$repo_dir/SOLO_Supervised_RFDETR/Stat_Dataset40x/QA-2025v1_TwoClass_OVR_V2_20261110-081112"
tiles="$work/tiles"
manifest="$work/ssl_training_specimens.csv"
public_weights="${STUDY3_PUBLIC_SSL_WEIGHTS:-}"
python_bin="${PYTHON_BIN:-python}"

case "$mode" in
  prepare)
    : "${CORRECTED_PATCH_ROOT:?Set this to the uploaded flat folder of corrected JPEG patches}"
    exec "$python_bin" "$repo_dir/40x_Study2_Preprocessing/tileDetectorDataset.py" \
      --source "$source_split" --image-root "$CORRECTED_PATCH_ROOT" --output "$tiles" \
      --tile-size 704 --min-overlap 0.25 --workers "${NATIVE704_TILE_WORKERS:-8}"
    ;;
esac

: "${DINOV3_REPO:?Run the DINOv3 initializer for this GPU job}"
require_public_weights() {
  if [[ ! -f "$public_weights" ]]; then
    echo "Set STUDY3_PUBLIC_SSL_WEIGHTS to the official DINOv3-S/16 LVD-1689M checkpoint." >&2
    exit 2
  fi
}

train_ssl() {
  local recipe="$1" epochs="$2" warmup="$3"
  local run="$work/ssl_$recipe" config="$work/ssl_$recipe.yaml" teacher
  local extra=()
  if [[ "$recipe" == adapt ]]; then
    extra=(--public-backbone-weights "$public_weights")
  fi
  if [[ -f "$run/completed.sha256" ]]; then
    sha256sum -c "$run/completed.sha256"
    return
  fi
  if ! mkdir "$run"; then
    echo "SSL output already exists without a completion record: $run. Inspect it before retrying." >&2
    exit 2
  fi
  "$python_bin" "$study_dir/make_ucloud_training_config.py" \
    --architecture vit_small --base-config "$study_dir/configs/dinov3_vits16_public_domain_adaptation_1gpu.yaml" \
    --manifest "$manifest" --output "$config" --gpus "$gpus" --batch-size-per-gpu "$batch" \
    --epochs "$epochs" --warmup-epochs "$warmup" --num-workers "${STUDY3_SSL_NUM_WORKERS:-8}" --seed 0
  DINOV3_RESUME=0 bash "$study_dir/run_official_dinov3_linux.sh" \
    "$DINOV3_REPO" "$config" "$manifest" "$ssl_root" "$run" "$gpus" "${extra[@]}"
  teacher="$("$python_bin" "$study_dir/find_latest_teacher_checkpoint.py" --run-dir "$run")"
  if [[ "$recipe" == adapt ]]; then
    sha256sum "$manifest" "$teacher" "$public_weights" > "$run/completed.sha256.part"
  else
    sha256sum "$manifest" "$teacher" > "$run/completed.sha256.part"
  fi
  printf '%s\n' "$teacher" > "$run/teacher_path.txt"
  mv -- "$run/completed.sha256.part" "$run/completed.sha256"
}

case "$mode" in
  ssl|ssl-scratch|ssl-adapt)
    [[ "$mode" == ssl-scratch ]] || require_public_weights
    ssl_root="${SSL_IMAGE_ROOT:-${IMAGE_ROOT:-}}"
    : "${ssl_root:?Set SSL_IMAGE_ROOT to the original full tile pool with one folder per specimen}"
    ssl_root="$(realpath "$ssl_root")"
    "$python_bin" "$study_dir/build_native704_ssl_manifest.py" \
      --dataset "$source_split" --image-root "$ssl_root" --output "$manifest"
    gpus="${STUDY3_SSL_GPUS:-$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')}"
    batch="${STUDY3_SSL_BATCH_PER_GPU:-32}"
    if [[ "$mode" != ssl-adapt ]]; then
      train_ssl scratch "${STUDY3_SSL_SCRATCH_EPOCHS:-20}" "${STUDY3_SSL_SCRATCH_WARMUP:-2}"
    fi
    if [[ "$mode" != ssl-scratch ]]; then
      train_ssl adapt "${STUDY3_SSL_ADAPT_EPOCHS:-10}" "${STUDY3_SSL_ADAPT_WARMUP:-1}"
    fi
    exit 0
    ;;
esac

if [[ ! -f "$tiles/tiling_summary.json" ]]; then
  echo "Run '$0 prepare' once with the shared OUTPUT_ROOT before starting detector training." >&2
  exit 2
fi
extra=()
if [[ "$mode" == public_ssl || "$mode" == public_domain_ssl ]]; then
  require_public_weights
  extra+=(--public-ssl-weights "$public_weights")
fi
if [[ "$mode" == own_data_ssl || "$mode" == public_domain_ssl ]]; then
  recipe=scratch
  [[ "$mode" != public_domain_ssl ]] || recipe=adapt
  run="$work/ssl_$recipe"
  sha256sum -c "$run/completed.sha256"
  teacher="$(cat "$run/teacher_path.txt")"
  if [[ "$recipe" == adapt ]] && ! grep -Fxq "$(sha256sum "$public_weights" | cut -d' ' -f1)  $public_weights" "$run/completed.sha256"; then
    echo "Public checkpoint differs from the one used for domain adaptation." >&2
    exit 2
  fi
  extra+=(--ssl-checkpoint "$teacher")
fi
echo "[Native704] Starting $mode detector with $python_bin"
echo "[Native704] Shared tiles: $tiles"
exec "$python_bin" -u "$study_dir/run_detection_experiments.py" \
  --config "$study_dir/detection_native704_ucloud_config.json" --arms "$mode" \
  --dinov3-repo "$DINOV3_REPO" --dataset-dir "$tiles" --image-root "$tiles" \
  --output-root "$work/detection" "${extra[@]}" --train
