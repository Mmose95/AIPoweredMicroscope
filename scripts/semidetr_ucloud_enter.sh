#!/usr/bin/env bash
set -euo pipefail

# Enter (or run within) a prepared Semi-DETR environment on UCloud.
# Usage:
#   scripts/semidetr_ucloud_enter.sh
#   scripts/semidetr_ucloud_enter.sh python -V

WORK_PROJECTS_DIR="${WORK_PROJECTS_DIR:-/work/projects/myproj}"
SEMIDETR_PARENT="${SEMIDETR_PARENT:-$WORK_PROJECTS_DIR/_external_SemiDETR_clean}"
SEMIDETR_REPO_DIR="${SEMIDETR_REPO_DIR:-$SEMIDETR_PARENT/Semi-DETR}"
CONDA_BASE="${CONDA_BASE:-/work/CondaEnv/miniconda3}"
SEMIDETR_ENV_NAME="${SEMIDETR_ENV_NAME:-semidetr}"
SEMIDETR_ENV_PATH="${SEMIDETR_ENV_PATH:-$CONDA_BASE/envs/$SEMIDETR_ENV_NAME}"

log() {
  printf "[SemiDETR enter] %s\n" "$*"
}

if [[ ! -x "$CONDA_BASE/bin/conda" ]]; then
  log "Conda not found at $CONDA_BASE."
  log "Run: bash scripts/semidetr_ucloud_bootstrap.sh"
  exit 1
fi

if [[ ! -d "$SEMIDETR_REPO_DIR" ]]; then
  log "Repo not found at $SEMIDETR_REPO_DIR."
  log "Run: bash scripts/semidetr_ucloud_bootstrap.sh"
  exit 1
fi

if [[ ! -d "$SEMIDETR_ENV_PATH" ]]; then
  log "Env not found at $SEMIDETR_ENV_PATH."
  log "Run: bash scripts/semidetr_ucloud_bootstrap.sh"
  exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$SEMIDETR_ENV_PATH"
cd "$SEMIDETR_REPO_DIR"

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

log "Activated: $CONDA_PREFIX"
log "Repo: $SEMIDETR_REPO_DIR"
exec "${SHELL:-bash}" -i
