#!/usr/bin/env bash
# Run in a GPU job initialized with init_native704_ucloud.sh.
set -Eeuo pipefail

study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$study_dir")"
source "$repo_dir/dinov3_ucloud_env.sh"
: "${OUTPUT_ROOT:?The UCloud initializer must set OUTPUT_ROOT}"
: "${PYTHON_BIN:?The UCloud initializer must set PYTHON_BIN}"

user_mount="$(dirname "$OUTPUT_ROOT")"
images="${SSL_IMAGE_ROOT:-$user_mount/40xSSLTrainingPatches}"
work="$OUTPUT_ROOT/Native704_20261110"
manifest="$work/ssl_training_specimens.csv"
dataset="$repo_dir/SOLO_Supervised_RFDETR/Stat_Dataset40x/QA-2025v1_TwoClass_OVR_V2_20261110-081112"
output="${DINOV2_SSL_OUTPUT:-$work/ssl_dinov2_small16_scratch}"
if [[ "${1:-}" == "--smoke" ]]; then
  output="${DINOV2_SSL_OUTPUT:-$work/ssl_dinov2_small16_smoke}"
fi

"$PYTHON_BIN" "$study_dir/build_native704_ssl_manifest.py" \
  --dataset "$dataset" --image-root "$images" --output "$manifest"
exec "$PYTHON_BIN" -u "$study_dir/train_dinov2_small_ssl.py" \
  --manifest "$manifest" --images "$images" --output "$output" \
  --batch-size "${STUDY3_SSL_BATCH_PER_GPU:-32}" --epochs "${STUDY3_SSL_SCRATCH_EPOCHS:-20}" \
  --warmup "${STUDY3_SSL_SCRATCH_WARMUP:-2}" --workers "${STUDY3_SSL_NUM_WORKERS:-8}" "$@"
