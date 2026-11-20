#!/usr/bin/env bash
# Match the 40-epoch, seed-0 random encoder control.
set -Eeuo pipefail
study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$study_dir")/dinov3_ucloud_env.sh"
work="$OUTPUT_ROOT/Native704_20261110"
experiment="${NATIVE_D_ROOT:-$work/ssl_native_d_768_384_v1}"
teacher="$experiment/training/teacher_epoch_030.pth"
dataset="$work/full_native2528"
output="$experiment/full_finetune_large_epoch030_40ep"
for file in "$teacher" "$dataset/train/_annotations.coco.json" "$dataset/valid/_annotations.coco.json"; do
  [[ -s "$file" ]] || { echo "Missing input: $file" >&2; exit 1; }
done
[[ ! -e "$output" ]] || { echo "Output already exists: $output" >&2; exit 1; }
weights="${STUDY3_STOCK_RFDETR_WEIGHTS:-${TMPDIR:-/tmp}/rf-detr-large-2026.pth}"
digest=0f4e20e19a99c0f8a62b5685f57f6c8b5c371c59081feda6752a0561a79ccf38
if [[ ! -f "$weights" ]]; then
  temporary="$(mktemp "${weights}.XXXXXX")"
  trap 'rm -f -- "$temporary"' EXIT
  curl --fail --location --output "$temporary" https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth
  printf '%s  %s\n' "$digest" "$temporary" | sha256sum -c -
  mv -- "$temporary" "$weights"
  trap - EXIT
fi
printf '%s  %s\n' "$digest" "$weights" | sha256sum -c -
visible_devices="${CUDA_VISIBLE_DEVICES-0}"
export CUDA_VISIBLE_DEVICES="${visible_devices%%,*}"
exec "$PYTHON_BIN" -u "$study_dir/probe_dinov2_teachers.py" \
  --worker --full-finetune --epochs 40 --teacher "$teacher" \
  --weights "$weights" --dataset "$dataset" --output "$output"
