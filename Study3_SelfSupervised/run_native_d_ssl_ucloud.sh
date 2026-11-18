#!/usr/bin/env bash
# Native-resolution weighted/guided D experiment; existing jobs are unchanged.
set -Eeuo pipefail
study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(dirname "$study_dir")"
source "$repo_dir/dinov3_ucloud_env.sh"
: "${OUTPUT_ROOT:?Initialize the UCloud job with init_native704_ucloud.sh}"
: "${PYTHON_BIN:?Missing initialized Python runtime}"
mode="${1:-benchmark}"
if (($#)); then shift; fi
work="$OUTPUT_ROOT/Native704_20261110"
experiment="${NATIVE_D_ROOT:-$work/ssl_native_d_768_384_v1}"
images="${SSL_IMAGE_ROOT:-$(dirname "$OUTPUT_ROOT")/40xSSLTrainingPatches}"
manifest="$experiment/training_manifest.csv"
weights="$experiment/sampling_weights.csv"
dataset="$repo_dir/SOLO_Supervised_RFDETR/Stat_Dataset40x/QA-2025v1_TwoClass_OVR_V2_20261110-081112"
gpus="${DINOV2_SSL_GPUS:-2}"
batch="${DINOV2_SSL_GLOBAL_BATCH:-8}"
epochs="${NATIVE_D_EPOCHS:-50}"
workers="${STUDY3_NUM_WORKERS:-4}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NO_ALBUMENTATIONS_UPDATE=1
if [[ "$mode" == prepare ]]; then
  "$PYTHON_BIN" "$study_dir/build_native704_ssl_manifest.py" --dataset "$dataset" --image-root "$images" --output "$manifest"
  exec "$PYTHON_BIN" "$study_dir/native_d_ssl.py" --manifest "$manifest" --images "$images" --output "$weights" --workers 8
fi
if [[ ! -f "$weights" ]]; then
  echo "First run: bash $study_dir/run_native_d_ssl_ucloud.sh prepare" >&2; exit 2
fi
if [[ "$mode" == probe ]]; then
  exec "$PYTHON_BIN" -u "$study_dir/screen_native_d_ssl.py" --work "$work" --experiment "$experiment" "$@"
fi
command=("$PYTHON_BIN" -u)
if [[ "$gpus" != 1 ]]; then command+=(-m torch.distributed.run --standalone --nproc-per-node "$gpus"); fi
extra=()
case "$mode" in
  benchmark) output="$experiment/benchmark-g${gpus}-b${batch}"; extra+=(--benchmark-warmup 10 --benchmark-steps 40) ;;
  train|resume)
    "$PYTHON_BIN" - "$experiment/benchmark-g${gpus}-b${batch}/benchmark.json" "$gpus" "$batch" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.is_file():raise SystemExit('Complete the matching GPU/batch benchmark before training.')
b=json.loads(p.read_text())
if b['gpus']!=int(sys.argv[2]) or b['global_batch']!=int(sys.argv[3]) or b['view_recipe']!='native-d':
    raise SystemExit('Benchmark configuration mismatch')
print('Feasibility check passed:',b,flush=True)
PY
    output="$experiment/training"; extra+=(--save-state)
    if [[ "$mode" == resume ]]; then extra+=(--resume); fi ;;
  *) echo "Usage: $0 {prepare|benchmark|train|resume|probe}" >&2; exit 2 ;;
esac
exec "${command[@]}" "$study_dir/train_dinov2_small_ssl.py" \
  --manifest "$manifest" --images "$images" --output "$output" \
  --view-recipe native-d --sampling-weights "$weights" --local-crops 8 \
  --batch-size "$batch" --epochs "$epochs" --warmup 5 --workers "$workers" \
  --teacher-every 10 "${extra[@]}" "$@"
