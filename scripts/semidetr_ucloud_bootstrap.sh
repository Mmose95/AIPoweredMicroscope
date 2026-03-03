#!/usr/bin/env bash
set -euo pipefail

# Idempotent Semi-DETR setup for UCloud.
# First run: installs conda (if needed), clones Semi-DETR, prepares env, installs modules, compiles ops.
# Later runs: reuses everything and exits quickly.

WORK_PROJECTS_DIR="${WORK_PROJECTS_DIR:-/work/projects/myproj}"
SEMIDETR_PARENT="${SEMIDETR_PARENT:-$WORK_PROJECTS_DIR/_external_SemiDETR_clean}"
SEMIDETR_REPO_DIR="${SEMIDETR_REPO_DIR:-$SEMIDETR_PARENT/Semi-DETR}"
SEMIDETR_REPO_URL="${SEMIDETR_REPO_URL:-https://github.com/JCZ404/Semi-DETR.git}"
SEMIDETR_REPO_REF="${SEMIDETR_REPO_REF:-main}"

CONDA_BASE="${CONDA_BASE:-/work/CondaEnv/miniconda3}"
SEMIDETR_ENV_NAME="${SEMIDETR_ENV_NAME:-semidetr}"
SEMIDETR_ENV_PATH="${SEMIDETR_ENV_PATH:-$CONDA_BASE/envs/$SEMIDETR_ENV_NAME}"

SEMIDETR_ENV_TAR_URL="${SEMIDETR_ENV_TAR_URL:-https://drive.google.com/file/d/1XoaMtMMVW4_qUGHXEOlEnjnOUaapyWwA/view?usp=drive_link}"
SEMIDETR_CACHE_DIR="${SEMIDETR_CACHE_DIR:-$SEMIDETR_PARENT/cache}"
SEMIDETR_ENV_TAR_PATH="${SEMIDETR_ENV_TAR_PATH:-$SEMIDETR_CACHE_DIR/semidetr_miniconda_cuda12.1_torch1.9.0+cu111_mmcv-full1.3.16.tar}"

SEMIDETR_UPDATE_REPO="${SEMIDETR_UPDATE_REPO:-0}"   # 1 => git pull when repo already exists
SEMIDETR_FORCE_SETUP="${SEMIDETR_FORCE_SETUP:-0}"   # 1 => rerun editable install + CUDA ops build

log() {
  printf "[SemiDETR setup] %s\n" "$*"
}

ensure_conda() {
  if [[ -x "$CONDA_BASE/bin/conda" ]]; then
    log "Conda found at $CONDA_BASE"
    return
  fi

  mkdir -p "$(dirname "$CONDA_BASE")"
  local installer="$SEMIDETR_CACHE_DIR/miniconda.sh"
  mkdir -p "$SEMIDETR_CACHE_DIR"
  if [[ ! -f "$installer" ]]; then
    log "Downloading Miniconda installer..."
    wget -O "$installer" "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  fi
  log "Installing Miniconda to $CONDA_BASE ..."
  bash "$installer" -b -p "$CONDA_BASE"
}

ensure_repo() {
  mkdir -p "$SEMIDETR_PARENT"
  if [[ -d "$SEMIDETR_REPO_DIR/.git" ]]; then
    log "Semi-DETR repo already exists: $SEMIDETR_REPO_DIR"
    if [[ "$SEMIDETR_UPDATE_REPO" == "1" ]]; then
      log "Updating repo (git pull --ff-only)..."
      git -C "$SEMIDETR_REPO_DIR" pull --ff-only
    fi
  else
    log "Cloning Semi-DETR into $SEMIDETR_REPO_DIR ..."
    git clone "$SEMIDETR_REPO_URL" "$SEMIDETR_REPO_DIR"
    git -C "$SEMIDETR_REPO_DIR" checkout "$SEMIDETR_REPO_REF"
  fi
}

activate_conda() {
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
}

ensure_env_from_tar() {
  if [[ -d "$SEMIDETR_ENV_PATH" ]]; then
    log "Environment already exists: $SEMIDETR_ENV_PATH"
    return
  fi

  mkdir -p "$SEMIDETR_CACHE_DIR" "$CONDA_BASE/envs"

  if [[ ! -f "$SEMIDETR_ENV_TAR_PATH" ]]; then
    log "Downloading official Semi-DETR env tar..."
    "$CONDA_BASE/bin/python" -m pip install --quiet gdown
    "$CONDA_BASE/bin/python" -m gdown --fuzzy "$SEMIDETR_ENV_TAR_URL" -O "$SEMIDETR_ENV_TAR_PATH"
  fi

  log "Extracting env tar to $CONDA_BASE/envs ..."
  tar -xf "$SEMIDETR_ENV_TAR_PATH" -C "$CONDA_BASE/envs"

  if [[ ! -d "$SEMIDETR_ENV_PATH" ]]; then
    local detected
    detected="$(tar -tf "$SEMIDETR_ENV_TAR_PATH" | head -n 1 | cut -d/ -f1 || true)"
    if [[ -n "$detected" && -d "$CONDA_BASE/envs/$detected" ]]; then
      SEMIDETR_ENV_PATH="$CONDA_BASE/envs/$detected"
      log "Detected extracted env path: $SEMIDETR_ENV_PATH"
    fi
  fi

  if [[ ! -d "$SEMIDETR_ENV_PATH" ]]; then
    log "ERROR: Could not locate extracted env directory."
    log "Set SEMIDETR_ENV_PATH explicitly and rerun."
    exit 1
  fi
}

install_project_packages() {
  local env_tag
  env_tag="$(basename "$SEMIDETR_ENV_PATH")"
  local marker="$SEMIDETR_REPO_DIR/.semidetr_setup_${env_tag}.ok"
  if [[ -f "$marker" && "$SEMIDETR_FORCE_SETUP" != "1" ]]; then
    log "Editable installs and CUDA ops already built ($marker)"
    return
  fi

  conda activate "$SEMIDETR_ENV_PATH"
  log "Active env: $CONDA_PREFIX"

  cd "$SEMIDETR_REPO_DIR"
  git submodule update --init --recursive || true

  log "Installing mmdetection (editable)..."
  cd thirdparty/mmdetection
  python -m pip install -r requirements/build.txt
  python -m pip install -e .

  log "Installing Semi-DETR package (editable)..."
  cd ../..
  python -m pip install -e .

  log "Compiling deformable attention CUDA ops..."
  cd detr_od/models/utils/ops
  python setup.py build install
  python test.py

  cd "$SEMIDETR_REPO_DIR"
  touch "$marker"
  log "Setup marker written: $marker"
}

verify_install() {
  conda activate "$SEMIDETR_ENV_PATH"
  cd "$SEMIDETR_REPO_DIR"
  python - <<'PY'
import torch
import mmcv
import mmdet
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "gpu:", torch.cuda.is_available())
print("mmcv:", mmcv.__version__)
print("mmdet:", mmdet.__version__)
PY
}

main() {
  log "Starting bootstrap"
  ensure_conda
  activate_conda
  ensure_repo
  ensure_env_from_tar
  install_project_packages
  verify_install
  log "Done."
  log "Next jobs: run scripts/semidetr_ucloud_enter.sh"
}

main "$@"
