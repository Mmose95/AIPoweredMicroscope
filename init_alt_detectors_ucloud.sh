#!/usr/bin/env bash
set -Eeuo pipefail

# UCloud notebook-job initializer for the alternative detector benchmark.
# This is intentionally separate from the RF-DETR/aipowmic init so the
# dependencies for DEIMv2, EdgeCrafter/ECDet, and D-FINE do not touch the
# environment that produced the current RF-DETR best results.

CONDA_BIN="${CONDA_BIN:-/work/CondaEnv/miniconda3/bin/conda}"
ENV_NAME="${ALTDET_ENV_NAME:-altdet311}"
PYTHON_VERSION="${ALTDET_PYTHON_VERSION:-3.11}"
PROJECT_ROOT="${PROJECT_ROOT:-/work/projects}"
PROJECT_NAME="${PROJECT_NAME:-myproj}"
REPO_URL="${REPO_URL:-https://github.com/Mmose95/AIPoweredMicroscope}"
SCRIPT_VERSION="2026-06-22-clone-first"
INIT_LOG="${INIT_LOG:-/work/projects/init_alt_detectors_ucloud.log}"

mkdir -p "$(dirname "${INIT_LOG}")"
exec > >(tee -a "${INIT_LOG}") 2>&1

echo "[InitAlt] Starting alternative-detector UCloud init"
echo "[InitAlt] SCRIPT_VERSION=${SCRIPT_VERSION}"
echo "[InitAlt] Log: ${INIT_LOG}"
echo "[InitAlt] ENV_NAME=${ENV_NAME}"

# 1) Get code first. If later env setup fails, the repo should still be visible.
mkdir -p "${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

PROJECT_DIR="${PROJECT_ROOT}/${PROJECT_NAME}"

if [ ! -d "${PROJECT_DIR}/.git" ]; then
  echo "[InitAlt] Cloning repo into ${PROJECT_DIR} ..."
  git clone "${REPO_URL}" "${PROJECT_DIR}"
else
  echo "[InitAlt] Pulling latest in ${PROJECT_DIR} ..."
  (cd "${PROJECT_DIR}" && git pull --ff-only || true)
fi

echo "[InitAlt] Project directory listing:"
ls -la "${PROJECT_ROOT}" || true

# 2) Load conda and create/activate isolated env.
if [ ! -x "${CONDA_BIN}" ]; then
  echo "[InitAlt][ERROR] Could not find conda at ${CONDA_BIN}" >&2
  exit 1
fi

eval "$("${CONDA_BIN}" shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[InitAlt] Creating conda env ${ENV_NAME} with Python ${PYTHON_VERSION} ..."
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
else
  echo "[InitAlt] Conda env ${ENV_NAME} already exists."
fi

conda activate "${ENV_NAME}"
echo "[InitAlt] Python: $(which python)"
python --version || true

# Register the Jupyter kernel early so it is visible even if a later package
# install fails. Re-running this is harmless.
python -m pip install --upgrade pip setuptools wheel ipykernel
python -m ipykernel install --user \
  --name "${ENV_NAME}" \
  --display-name "Python (${ENV_NAME})"

# 3) Detect AAU vs SDU user base.
if compgen -G "/work/Member Files:*" >/dev/null; then
  USER_BASE_DIR="$(basename "$(ls -d /work/Member\ Files:* | head -n1)")"
else
  USER_BASE_DIR="$(basename "$(ls -d /work/*#* | head -n1)")"
fi

export USER_BASE_DIR
echo "[InitAlt] USER_BASE_DIR=${USER_BASE_DIR}"

cd "${PROJECT_DIR}"

# 4) CUDA-aware PyTorch install.
# For B200 jobs, prefer a CUDA wheel index. Do not silently fall back to CPU
# unless ALTDET_ALLOW_CPU_TORCH=1 is explicitly set.
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

if ! python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1 || [ "${REINSTALL_TORCH:-0}" = "1" ]; then
  echo "[InitAlt] Installing CUDA PyTorch for this node ..."
  CUDA_VER=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VER="$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}' | cut -d. -f1,2 | head -n1 || true)"
  fi

  TORCH_IDX="${ALTDET_TORCH_INDEX_URL:-$(choose_torch_index "${CUDA_VER}")}"
  if [ -z "${TORCH_IDX}" ]; then
    if [ "${ALTDET_ALLOW_CPU_TORCH:-0}" = "1" ]; then
      TORCH_IDX="https://download.pytorch.org/whl/cpu"
    else
      echo "[InitAlt][ERROR] Could not choose a CUDA torch index from CUDA_VER='${CUDA_VER}'." >&2
      echo "[InitAlt][ERROR] Set ALTDET_TORCH_INDEX_URL, for example https://download.pytorch.org/whl/cu128" >&2
      exit 1
    fi
  fi

  echo "[InitAlt] CUDA_VER=${CUDA_VER:-unknown}"
  echo "[InitAlt] Using PyTorch index: ${TORCH_IDX}"

  if [ -n "${ALTDET_TORCH_VERSION:-}" ] || [ -n "${ALTDET_TORCHVISION_VERSION:-}" ] || [ -n "${ALTDET_TORCHAUDIO_VERSION:-}" ]; then
    if [ -z "${ALTDET_TORCH_VERSION:-}" ] || [ -z "${ALTDET_TORCHVISION_VERSION:-}" ] || [ -z "${ALTDET_TORCHAUDIO_VERSION:-}" ]; then
      echo "[InitAlt][ERROR] To pin torch, set all three: ALTDET_TORCH_VERSION, ALTDET_TORCHVISION_VERSION, ALTDET_TORCHAUDIO_VERSION." >&2
      exit 1
    fi
    python -m pip install --upgrade \
      "torch==${ALTDET_TORCH_VERSION}" \
      "torchvision==${ALTDET_TORCHVISION_VERSION}" \
      "torchaudio==${ALTDET_TORCHAUDIO_VERSION}" \
      --index-url "${TORCH_IDX}"
  else
    python -m pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_IDX}"
  fi
else
  echo "[InitAlt] CUDA-enabled torch already present; skipping torch install."
fi

# 5) Common alt-detector packages. Repo-specific requirements are still handled
# by RunAltDetectors_DEIMv2_EdgeCrafter_DFINE_Ucloud.ipynb in non_torch mode.
echo "[InitAlt] Installing common detector packages ..."
python -m pip install --upgrade --no-cache-dir \
  packaging ninja cython \
  numpy==2.0.1 \
  ml_dtypes==0.2.0 \
  pycocotools PyYAML tensorboard \
  scipy==1.16.0 \
  calflops transformers tabulate \
  onnx==1.19.0 onnxruntime opencv-python \
  einops timm

# 6) Persistent defaults for the benchmark notebook.
export ALTDET_PYTHON="${CONDA_PREFIX}/bin/python"
export ALTDET_INSTALL_REQUIREMENTS_MODE="${ALTDET_INSTALL_REQUIREMENTS_MODE:-non_torch}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:/work/CondaEnv/mlflow/mlruns}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-AIPoweredMicroscope}"

cat > "${PROJECT_DIR}/alt_detector_env_ucloud.sh" <<EOF
export ALTDET_PYTHON="${ALTDET_PYTHON}"
export ALTDET_INSTALL_REQUIREMENTS_MODE="${ALTDET_INSTALL_REQUIREMENTS_MODE}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME}"
EOF

# 7) Final checks.
echo "[InitAlt] Torch check:"
python - <<'PY'
import sys
import torch

print(" python:", sys.executable)
print(" torch:", torch.__version__)
print(" cuda_available:", torch.cuda.is_available())
print(" cuda_runtime:", torch.version.cuda)
print(" gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print(" gpu0:", torch.cuda.get_device_name(0))
PY

echo "[InitAlt] Repo at: ${PROJECT_DIR}"
echo "[InitAlt] Kernel: Python (${ENV_NAME})"
echo "[InitAlt] Env file: ${PROJECT_DIR}/alt_detector_env_ucloud.sh"
echo "=================="
echo "==  Ready       =="
echo "=================="
