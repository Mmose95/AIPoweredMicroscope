#!/usr/bin/env bash
set -Eeuo pipefail

# UCloud job initializer for Study 3 DINOv3 self-supervised training.
# Intended for the Initialization field of a GPU-backed Jupyter/PyTorch job.

CONDA_BIN="${CONDA_BIN:-}"
ENV_NAME="${DINOV3_ENV_NAME:-phd-dinov3}"
ENV_DIR="${DINOV3_ENV_DIR:-/work/CondaEnv/envs/${ENV_NAME}}"
PROJECT_ROOT="${PROJECT_ROOT:-/work/projects}"
PROJECT_DIR="${PROJECT_DIR:-${PROJECT_ROOT}/myproj}"
DINOV3_REPO="${DINOV3_REPO:-${PROJECT_ROOT}/dinov3}"
PROJECT_REPO_URL="${PROJECT_REPO_URL:-https://github.com/Mmose95/AIPoweredMicroscope}"
DINOV3_REPO_URL="${DINOV3_REPO_URL:-https://github.com/facebookresearch/dinov3.git}"
DINOV3_COMMIT="${DINOV3_COMMIT:-6876159a11b4df116f30f667f8c9888617df0751}"
TORCH_VERSION="${DINOV3_TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${DINOV3_TORCHVISION_VERSION:-0.22.1}"
PYTHON_VERSION="${DINOV3_PYTHON_VERSION:-3.11}"

echo "[DINOv3 Init] Starting"

if [[ -z "$CONDA_BIN" ]]; then
  for candidate in \
    /work/CondaEnv/miniconda3/bin/conda \
    /opt/conda/bin/conda
  do
    if [[ -x "$candidate" ]]; then
      CONDA_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$CONDA_BIN" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
  echo "[DINOv3 Init][ERROR] Conda was not found under /work, /opt, or PATH." >&2
  exit 1
fi
echo "[DINOv3 Init] Conda: $CONDA_BIN"
eval "$("$CONDA_BIN" shell.bash hook)"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "[DINOv3 Init] Creating isolated environment: $ENV_DIR"
  mkdir -p "$(dirname "$ENV_DIR")"
  conda create -y -p "$ENV_DIR" -c conda-forge --override-channels "python=$PYTHON_VERSION" pip
fi
conda activate "$ENV_DIR"
python -m pip install --upgrade pip setuptools wheel
echo "[DINOv3 Init] Python: $(command -v python)"
python --version

# Detect the persistent AAU/SDU member mount.
if compgen -G "/work/Member Files:*" >/dev/null; then
  USER_BASE_PATH="$(find /work -maxdepth 1 -type d -name 'Member Files:*' -print -quit)"
else
  USER_BASE_PATH="$(find /work -maxdepth 1 -type d -name '*#*' -print -quit)"
fi
if [[ -z "${USER_BASE_PATH:-}" ]]; then
  # UCloud may mount selected folders directly below /work instead of exposing
  # the enclosing AAU/SDU member directory.
  USER_BASE_PATH="/work"
  USER_BASE_DIR=""
  echo "[DINOv3 Init] Using direct /work mounts"
else
  USER_BASE_DIR="$(basename "$USER_BASE_PATH")"
fi
export USER_BASE_DIR
echo "[DINOv3 Init] USER_BASE_PATH=$USER_BASE_PATH"
echo "[DINOv3 Init] USER_BASE_DIR=${USER_BASE_DIR:-<direct>}"

mkdir -p "$PROJECT_ROOT"
if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  echo "[DINOv3 Init] Cloning project into $PROJECT_DIR"
  git clone "$PROJECT_REPO_URL" "$PROJECT_DIR"
else
  echo "[DINOv3 Init] Updating project"
  git -C "$PROJECT_DIR" pull --ff-only || true
fi

if [[ ! -d "$DINOV3_REPO/.git" ]]; then
  echo "[DINOv3 Init] Cloning official DINOv3 into $DINOV3_REPO"
  git clone "$DINOV3_REPO_URL" "$DINOV3_REPO"
else
  echo "[DINOv3 Init] Official DINOv3 checkout already exists"
fi
echo "[DINOv3 Init] Selecting tested DINOv3 commit $DINOV3_COMMIT"
git -C "$DINOV3_REPO" fetch origin "$DINOV3_COMMIT"
git -C "$DINOV3_REPO" checkout --detach "$DINOV3_COMMIT"

choose_torch_index() {
  local cuda_version="${1:-}"
  case "$cuda_version" in
    13.*|12.8|12.9) echo "https://download.pytorch.org/whl/cu128" ;;
    12.6|12.7)      echo "https://download.pytorch.org/whl/cu126" ;;
    *)              echo "" ;;
  esac
}

CUDA_VERSION=""
if command -v nvidia-smi >/dev/null 2>&1; then
  # Standard drivers print "CUDA Version" while UCloud's newer B200/MIG
  # stack prints "CUDA UMD Version".
  CUDA_VERSION="$(
    nvidia-smi \
      | grep -oE 'CUDA (UMD )?Version: [0-9.]+' \
      | grep -oE '[0-9.]+$' \
      | cut -d. -f1,2 \
      | head -n1 \
      || true
  )"
fi
echo "[DINOv3 Init] Detected CUDA version: ${CUDA_VERSION:-unknown}"
TORCH_INDEX="${DINOV3_TORCH_INDEX_URL:-$(choose_torch_index "$CUDA_VERSION")}"
if [[ -z "$TORCH_INDEX" ]]; then
  echo "[DINOv3 Init][ERROR] No supported CUDA wheel selected for CUDA '$CUDA_VERSION'." >&2
  echo "Set DINOV3_TORCH_INDEX_URL explicitly; CPU fallback is intentionally disabled." >&2
  exit 1
fi

if ! python - <<PY
import torch
expected = "${TORCH_VERSION}"
raise SystemExit(0 if torch.__version__.split("+")[0] == expected and torch.cuda.is_available() else 1)
PY
then
  echo "[DINOv3 Init] Installing torch $TORCH_VERSION from $TORCH_INDEX"
  python -m pip install --upgrade \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
    --index-url "$TORCH_INDEX"
else
  echo "[DINOv3 Init] Requested CUDA-enabled PyTorch is already installed"
fi

echo "[DINOv3 Init] Installing official DINOv3 dependencies"
python -m pip install -r "$DINOV3_REPO/requirements.txt"
python -m pip install -e "$DINOV3_REPO"
python -m pip install --upgrade ipykernel
python -m ipykernel install --user \
  --name "$ENV_NAME" --display-name "Python ($ENV_NAME)" || true

# Prefer an explicit path. Otherwise check common storage layouts, then search.
if [[ -n "${DINOV3_IMAGE_ROOT:-}" ]]; then
  IMAGE_ROOT="$DINOV3_IMAGE_ROOT"
else
  IMAGE_ROOT=""
  for candidate in \
    "$USER_BASE_PATH/CellScanData/40x Input tiles for CVAT" \
    "$USER_BASE_PATH/40x Input tiles for CVAT" \
    "$USER_BASE_PATH/PHD/PhdData/Patologi afd. - Aalborg/40x Input tiles for CVAT"
  do
    if [[ -d "$candidate" ]]; then
      IMAGE_ROOT="$candidate"
      break
    fi
  done
  if [[ -z "$IMAGE_ROOT" ]]; then
    IMAGE_ROOT="$(find "$USER_BASE_PATH" -maxdepth 6 -type d -name '40x Input tiles for CVAT' -print -quit 2>/dev/null || true)"
  fi
fi
if [[ -z "$IMAGE_ROOT" || ! -d "$IMAGE_ROOT" ]]; then
  echo "[DINOv3 Init][ERROR] Could not locate '40x Input tiles for CVAT'." >&2
  echo "Set DINOV3_IMAGE_ROOT to the uploaded folder's absolute path." >&2
  exit 1
fi

STUDY3_DIR="$PROJECT_DIR/Study3_SelfSupervised"
FULL_MANIFEST="$STUDY3_DIR/manifests/ssl_pool_40x_9d8cb0d9ec7b.csv"
OUTPUT_ROOT="${DINOV3_OUTPUT_ROOT:-$USER_BASE_PATH/DINOv3_Study3_OUTPUT}"
mkdir -p "$OUTPUT_ROOT"

export DINOV3_REPO PROJECT_DIR STUDY3_DIR IMAGE_ROOT FULL_MANIFEST OUTPUT_ROOT
cat > "$PROJECT_DIR/dinov3_ucloud_env.sh" <<EOF
export USER_BASE_DIR=$(printf '%q' "$USER_BASE_DIR")
export DINOV3_REPO=$(printf '%q' "$DINOV3_REPO")
export PROJECT_DIR=$(printf '%q' "$PROJECT_DIR")
export STUDY3_DIR=$(printf '%q' "$STUDY3_DIR")
export IMAGE_ROOT=$(printf '%q' "$IMAGE_ROOT")
export FULL_MANIFEST=$(printf '%q' "$FULL_MANIFEST")
export OUTPUT_ROOT=$(printf '%q' "$OUTPUT_ROOT")
export DINOV3_ENV_DIR=$(printf '%q' "$ENV_DIR")
EOF

echo "[DINOv3 Init] Final verification"
python - <<'PY'
import sys
import torch
import dinov3
print(" python:", sys.executable)
print(" torch:", torch.__version__)
print(" cuda_available:", torch.cuda.is_available())
print(" cuda_runtime:", torch.version.cuda)
print(" gpu_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
for index in range(torch.cuda.device_count()):
    print(f" gpu{index}:", torch.cuda.get_device_name(index))
print(" DINOv3 import: passed")
PY

if [[ "${DINOV3_VALIDATE_DATA:-0}" == "1" ]]; then
  python "$STUDY3_DIR/validate_uploaded_ssl_pool.py" \
    --manifest "$FULL_MANIFEST" \
    --image-root "$IMAGE_ROOT" \
    --check-sizes
fi

echo "[DINOv3 Init] IMAGE_ROOT=$IMAGE_ROOT"
echo "[DINOv3 Init] OUTPUT_ROOT=$OUTPUT_ROOT"
echo "[DINOv3 Init] Environment file: $PROJECT_DIR/dinov3_ucloud_env.sh"
echo "=================="
echo "== DINOv3 ready =="
echo "=================="
