#!/usr/bin/env bash
set -euo pipefail

# Isolated environment for alternative detector benchmarking on UCloud/B200.
# This intentionally does not modify the RF-DETR/aipowmic environment.

CONDA_BASE="${CONDA_BASE:-/work/CondaEnv/miniconda3}"
ENV_NAME="${ALTDET_ENV_NAME:-altdet311}"
PYTHON_VERSION="${ALTDET_PYTHON_VERSION:-3.11}"

TORCH_VERSION="${ALTDET_TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${ALTDET_TORCHVISION_VERSION:-0.26.0}"
TORCHAUDIO_VERSION="${ALTDET_TORCHAUDIO_VERSION:-2.11.0}"
PYTORCH_INDEX_URL="${ALTDET_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  echo "Could not find conda at ${CONDA_BASE}." >&2
  echo "Set CONDA_BASE=/path/to/miniconda3 and rerun." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel

# Official PyTorch CUDA 12.8 wheels. Keep this explicit so third-party detector
# requirements cannot silently downgrade the B200-compatible torch stack.
python -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${PYTORCH_INDEX_URL}"

# Common non-framework packages used by DEIMv2, EdgeCrafter/ECDet, and D-FINE.
# The benchmark notebook still installs each repo's non-torch requirements.
python -m pip install \
  packaging ninja cython \
  numpy==2.0.1 \
  ml_dtypes==0.2.0 \
  pycocotools PyYAML tensorboard \
  scipy==1.16.0 \
  calflops transformers tabulate \
  onnx==1.19.0 onnxruntime opencv-python \
  einops timm

python - <<'PY'
import sys
import torch

print("python:", sys.executable)
print("python_version:", sys.version)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_runtime:", torch.version.cuda)
print("gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
PY

cat <<EOF

Use this interpreter in RunAltDetectors_DEIMv2_EdgeCrafter_DFINE_Ucloud.ipynb:

  export ALTDET_PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
  export ALTDET_INSTALL_REQUIREMENTS_MODE="non_torch"

Or set inside the notebook before running setup:

  os.environ["ALTDET_PYTHON"] = "${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
  os.environ["ALTDET_INSTALL_REQUIREMENTS_MODE"] = "non_torch"

EOF
