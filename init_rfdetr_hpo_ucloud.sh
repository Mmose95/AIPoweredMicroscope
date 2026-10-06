#!/usr/bin/env bash
set -Eeuo pipefail

# UCloud initializer for the focused RF-DETR Large HPO study.
# Safe to run again: the repository and conda environment are reused.

CONDA_BIN="${CONDA_BIN:-/work/CondaEnv/miniconda3/bin/conda}"
ENV_NAME="${RFDETR_ENV_NAME:-rfdetr180}"
ENV_DIR="${RFDETR_ENV_DIR:-/work/CondaEnv/envs/${ENV_NAME}}"
PROJECT_ROOT="${PROJECT_ROOT:-/work/projects}"
PROJECT_NAME="${PROJECT_NAME:-myproj}"
PROJECT_DIR="${PROJECT_ROOT}/${PROJECT_NAME}"
REPO_URL="${REPO_URL:-https://github.com/Mmose95/AIPoweredMicroscope}"
RFDETR_VERSION="${RFDETR_VERSION:-1.8.0}"
INIT_LOG="${INIT_LOG:-/work/projects/init_rfdetr_hpo_ucloud.log}"

mkdir -p "$(dirname "${INIT_LOG}")"
exec > >(tee -a "${INIT_LOG}") 2>&1

echo "[RFDETR init] Project: ${PROJECT_DIR}"
echo "[RFDETR init] Environment: ${ENV_DIR}"
echo "[RFDETR init] RF-DETR: ${RFDETR_VERSION}"

mkdir -p "${PROJECT_ROOT}"
if [ ! -d "${PROJECT_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${PROJECT_DIR}"
else
  (cd "${PROJECT_DIR}" && git pull --ff-only)
fi

if [ ! -x "${CONDA_BIN}" ]; then
  echo "[RFDETR init][ERROR] Conda was not found at ${CONDA_BIN}" >&2
  exit 1
fi

eval "$("${CONDA_BIN}" shell.bash hook)"
for channel in \
  "https://repo.anaconda.com/pkgs/main" \
  "https://repo.anaconda.com/pkgs/r"
do
  conda tos accept --override-channels --channel "${channel}" || true
done

if [ ! -x "${ENV_DIR}/bin/python" ]; then
  conda create -y -p "${ENV_DIR}" python=3.11
fi
conda activate "${ENV_DIR}"

python -m pip install --upgrade pip setuptools wheel

choose_torch_index() {
  local cuda_ver="${1:-}"
  case "${cuda_ver}" in
    13.*|12.9|12.8) echo "https://download.pytorch.org/whl/cu128" ;;
    12.6)           echo "https://download.pytorch.org/whl/cu126" ;;
    12.4)           echo "https://download.pytorch.org/whl/cu124" ;;
    12.1)           echo "https://download.pytorch.org/whl/cu121" ;;
    *)              echo "" ;;
  esac
}

if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  CUDA_VER=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER="$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}' | cut -d. -f1,2 | head -n1 || true)"
  fi
  TORCH_INDEX="${RFDETR_TORCH_INDEX_URL:-$(choose_torch_index "${CUDA_VER}")}"
  if [ -z "${TORCH_INDEX}" ]; then
    echo "[RFDETR init][ERROR] Could not select a CUDA PyTorch wheel for CUDA ${CUDA_VER:-unknown}." >&2
    echo "Set RFDETR_TORCH_INDEX_URL explicitly and rerun the initializer." >&2
    exit 1
  fi
  python -m pip install --upgrade torch torchvision --index-url "${TORCH_INDEX}"
fi

python -m pip install --upgrade --no-cache-dir \
  "rfdetr==${RFDETR_VERSION}" \
  pycocotools pillow tifffile pandas matplotlib tensorboard
python -m pip install --upgrade ipykernel
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

cat > "${PROJECT_DIR}/rfdetr_hpo_env_ucloud.sh" <<EOF
export RFDETR_PYTHON="${ENV_DIR}/bin/python"
export PROJECT_DIR="${PROJECT_DIR}/SOLO_Supervised_RFDETR"
export RFDETR_VERSION="${RFDETR_VERSION}"
EOF

python - <<'PY'
import importlib.metadata
import torch
from rfdetr import RFDETRLarge

print("rfdetr:", importlib.metadata.version("rfdetr"))
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
print("RFDETRLarge import: OK", RFDETRLarge)
PY

echo "[RFDETR init] Ready."
echo "[RFDETR init] Run script: ${PROJECT_DIR}/run_rfdetr_large_hpo_ucloud.sh"
echo "[RFDETR init] Environment file: ${PROJECT_DIR}/rfdetr_hpo_env_ucloud.sh"
