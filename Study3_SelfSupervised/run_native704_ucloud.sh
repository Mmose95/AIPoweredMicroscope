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
  prepare|prepare-full|prepare-full-10x-sampling|scratch|public_ssl|stock_pretrained|stock_pretrained_frozen|stock_pretrained_gentle|stock_pretrained_full|stock_pretrained_full_smoke|stock_pretrained_full_10x_sampling|public_ssl_full|public_ssl_full_smoke|ssl|ssl-scratch|ssl-adapt|own_data_ssl|public_domain_ssl) ;;
  *) echo "Usage: $0 {prepare|prepare-full|prepare-full-10x-sampling|scratch|public_ssl|stock_pretrained|stock_pretrained_frozen|stock_pretrained_gentle|stock_pretrained_full|stock_pretrained_full_smoke|stock_pretrained_full_10x_sampling|public_ssl_full|public_ssl_full_smoke|ssl|ssl-scratch|ssl-adapt|own_data_ssl|public_domain_ssl}" >&2; exit 2 ;;
esac

: "${OUTPUT_ROOT:?Source the UCloud environment or set OUTPUT_ROOT to persistent storage}"
work="$OUTPUT_ROOT/Native704_20261110"
source_split="$repo_dir/SOLO_Supervised_RFDETR/Stat_Dataset40x/QA-2025v1_TwoClass_OVR_V2_20261110-081112"
tiles="$work/tiles"
manifest="$work/ssl_training_specimens.csv"
public_weights="${STUDY3_PUBLIC_SSL_WEIGHTS:-}"
python_bin="${PYTHON_BIN:-python}"
detection_output="${NATIVE704_DETECTION_OUTPUT:-$work/detection_local_runtime}"
sampled_full="$work/full_native2528_10x_sampling"

case "$mode" in
  prepare|prepare-full|prepare-full-10x-sampling)
    : "${CORRECTED_PATCH_ROOT:?Set this to the uploaded flat folder of corrected JPEG patches}"
    geometry=(--tile-size 704 --min-overlap 0.25)
    if [[ "$mode" == prepare-full* ]]; then
      tiles="$work/full_native2528"
      geometry=(--tile-size 2528 --pad-only)
    fi
    if [[ "$mode" == prepare-full-10x-sampling ]]; then
      tiles="$sampled_full"
      geometry+=(--sampling-size 640 640)
    fi
    exec "$python_bin" "$repo_dir/40x_Study2_Preprocessing/tileDetectorDataset.py" \
      --source "$source_split" --image-root "$CORRECTED_PATCH_ROOT" --output "$tiles" \
      "${geometry[@]}" --workers "${NATIVE704_TILE_WORKERS:-8}"
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

config="$study_dir/detection_native704_ucloud_config.json"
prepare_mode=prepare
extra=()
case "$mode" in
  stock_pretrained_full_10x_sampling)
    tiles="$sampled_full"
    config="$study_dir/detection_native2528_ucloud_config.json"
    prepare_mode=prepare-full-10x-sampling
    detection_output="$detection_output/10x_sampling"
    mode=stock_pretrained
    ;;
  stock_pretrained_frozen|stock_pretrained_gentle)
    extra+=(--stock-policy "${mode#stock_pretrained_}")
    mode=stock_pretrained
    ;;
  public_ssl_full*|stock_pretrained_full*)
    tiles="$work/full_native2528"
    config="$study_dir/detection_native2528_ucloud_config.json"
    prepare_mode=prepare-full
    if [[ "$mode" == *_smoke ]]; then
      extra+=(--smoke-test)
    fi
    mode="${mode%%_full*}"
    if [[ "$mode" == public_ssl ]]; then
      extra+=(--progress-bar tqdm)
    fi
    ;;
esac
if [[ ! -f "$tiles/tiling_summary.json" ]]; then
  echo "Run '$0 $prepare_mode' once with the shared OUTPUT_ROOT before starting detector training." >&2
  exit 2
fi
if [[ "$mode" == stock_pretrained ]]; then
  stock_weights="${STUDY3_STOCK_RFDETR_WEIGHTS:-${TMPDIR:-/tmp}/rf-detr-large-2026.pth}"
  expected_md5=5cb72153541cbcb9aa6efa26222acc75
  if [[ ! -f "$stock_weights" && -z "${STUDY3_STOCK_RFDETR_WEIGHTS:-}" ]]; then
    echo "[Native704] Downloading published RF-DETR Large checkpoint"
    curl --fail --location --output "$stock_weights.part" \
      https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth
    printf '%s  %s\n' "$expected_md5" "$stock_weights.part" | md5sum --check --status
    mv -- "$stock_weights.part" "$stock_weights"
  fi
  if ! printf '%s  %s\n' "$expected_md5" "$stock_weights" | md5sum --check --status; then
    echo "[Native704][ERROR] Missing checkpoint or checksum mismatch: $stock_weights" >&2
    exit 2
  fi
  extra+=(--stock-weights "$stock_weights" --progress-bar tqdm)
fi
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
  --config "$config" --arms "$mode" \
  --dinov3-repo "$DINOV3_REPO" --dataset-dir "$tiles" --image-root "$tiles" \
  --output-root "$detection_output" "${extra[@]}" --train
