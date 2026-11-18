#!/usr/bin/env bash
# Six independent, unfrozen Large detector runs; existing frozen probes are untouched.
set -Eeuo pipefail
study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$study_dir")/dinov3_ucloud_env.sh"
work="$OUTPUT_ROOT/Native704_20261110"
weights="${STUDY3_STOCK_RFDETR_WEIGHTS:-${TMPDIR:-/tmp}/rf-detr-large-2026.pth}"
if [[ ! -f "$weights" ]]; then
  curl --fail --location --output "$weights.part" \
    https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth
  mv -- "$weights.part" "$weights"
fi
exec "$PYTHON_BIN" -u "$study_dir/probe_dinov2_teachers.py" \
  --work "$work" --weights "$weights" --full-finetune --epochs 10 \
  --output "${DINOV2_SCREEN_OUTPUT:-$work/dinov2_teacher_screen_large_full_10ep}" "$@"
