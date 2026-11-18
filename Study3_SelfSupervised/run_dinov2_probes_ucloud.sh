#!/usr/bin/env bash
set -Eeuo pipefail
study_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$study_dir")/dinov3_ucloud_env.sh"
work="$OUTPUT_ROOT/Native704_20261110"
exec "$PYTHON_BIN" -u "$study_dir/probe_dinov2_teachers.py" \
  --work "$work" --output "${DINOV2_PROBE_OUTPUT:-$work/dinov2_teacher_probes_150}" "$@"
