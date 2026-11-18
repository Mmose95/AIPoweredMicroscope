#!/usr/bin/env bash
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
exec "$PYTHON_BIN" -u "$study_dir/extend_dinov2_screen.py" \
  --work "$work" --weights "$weights" \
  --source "${DINOV2_SCREEN_SOURCE:-$work/dinov2_teacher_screen_large_full_10ep}" \
  --output "${DINOV2_SCREEN40_OUTPUT:-$work/dinov2_teacher_screen_large_full_40ep}" "$@"
