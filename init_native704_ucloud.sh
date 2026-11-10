#!/usr/bin/env bash
# UCloud initializer for the Native704 four-arm jobs.
set -Eeuo pipefail

RUNTIME_ROOT="${NATIVE704_RUNTIME_ROOT:-/tmp/native704-runtime}"
ENV_DIR="$RUNTIME_ROOT/env"
PROJECT_DIR="${PROJECT_DIR:-/work/projects/myproj}"
DINOV3_REPO="$RUNTIME_ROOT/dinov3"
PROJECT_URL="${PROJECT_REPO_URL:-https://github.com/Mmose95/AIPoweredMicroscope.git}"
DINOV3_URL="${DINOV3_REPO_URL:-https://github.com/facebookresearch/dinov3.git}"
DINOV3_COMMIT="${DINOV3_COMMIT:-6876159a11b4df116f30f667f8c9888617df0751}"

OUTPUT_ROOT="${DINOV3_OUTPUT_ROOT:-${OUTPUT_ROOT:-}}"
if [[ -z "$OUTPUT_ROOT" ]]; then
  for mount in /work/Member\ Files:* /work/*#*; do
    if [[ -d "$mount" ]]; then
      OUTPUT_ROOT="$mount/DINOv3_Study3_OUTPUT"
      break
    fi
  done
fi
: "${OUTPUT_ROOT:?Set DINOV3_OUTPUT_ROOT to a persistent directory shared by all jobs}"
mkdir -p -- "$OUTPUT_ROOT" "$(dirname -- "$PROJECT_DIR")" "$(dirname -- "$DINOV3_REPO")"

echo "[Native704 Init] Preparing experiment checkout: $PROJECT_DIR"
if [[ -d "$PROJECT_DIR/.git" ]]; then
  git -C "$PROJECT_DIR" pull --ff-only
else
  git clone --depth 1 "$PROJECT_URL" "$PROJECT_DIR"
fi
if [[ ! -f "$PROJECT_DIR/Study3_SelfSupervised/run_native704_ucloud.sh" ]]; then
  echo "[Native704 Init][ERROR] The checkout does not contain the Native704 launcher." >&2
  exit 1
fi

echo "[Native704 Init] Preparing pinned DINOv3 checkout: $DINOV3_REPO"
if [[ ! -d "$DINOV3_REPO/.git" ]]; then
  git init --quiet "$DINOV3_REPO"
fi
if ! git -C "$DINOV3_REPO" cat-file -e "${DINOV3_COMMIT}^{commit}" 2>/dev/null; then
  git -C "$DINOV3_REPO" fetch --depth 1 "$DINOV3_URL" "$DINOV3_COMMIT"
fi
if [[ "$(git -C "$DINOV3_REPO" rev-parse --verify HEAD 2>/dev/null || true)" != "$DINOV3_COMMIT" ]]; then
  git -C "$DINOV3_REPO" checkout --detach "$DINOV3_COMMIT"
fi

echo "[Native704 Init] Setting up the job-local Python environment: $ENV_DIR"
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  bootstrap="${NATIVE704_BOOTSTRAP_PYTHON:-}"
  if [[ -z "$bootstrap" ]]; then
    for candidate in /opt/conda/bin/python3.11 /usr/bin/python3.11 /usr/local/bin/python3.11; do
      if [[ -x "$candidate" ]]; then
        bootstrap="$candidate"
        break
      fi
    done
  fi
  if [[ -n "$bootstrap" && "$(readlink -f -- "$bootstrap")" != /work/* ]]; then
    "$bootstrap" -m venv --copies "$ENV_DIR"
  elif [[ -x /opt/conda/bin/conda ]]; then
    /opt/conda/bin/conda create -y -p "$ENV_DIR" python=3.11 pip
  else
    echo "[Native704 Init][ERROR] Select a job image with Python 3.11 or local Conda." >&2
    exit 1
  fi
fi

python_bin="$ENV_DIR/bin/python"
torch_index="${DINOV3_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
marker="$RUNTIME_ROOT/ready-torch2.7.1-rfdetr1.9.1-${DINOV3_COMMIT}"
if [[ ! -f "$marker" ]]; then
  echo "[Native704 Init] Installing pinned CUDA PyTorch and training dependencies"
  "$python_bin" -m pip install --index-url "$torch_index" torch==2.7.1 torchvision==0.22.1
  constraints="$RUNTIME_ROOT/constraints.txt"
  printf '%s\n' torch==2.7.1 torchvision==0.22.1 'rfdetr==1.9.1' > "$constraints"
  "$python_bin" -m pip install -c "$constraints" -r "$DINOV3_REPO/requirements.txt" \
    -e "$DINOV3_REPO" 'rfdetr[train,augment]==1.9.1'
fi
echo "[Native704 Init] Verifying PyTorch, DINOv3, RF-DETR and CUDA"
"$python_bin" -u - <<'PY'
import torch
import dinov3
from rfdetr import RFDETRLarge

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the job-local environment")
print(f"PyTorch {torch.__version__}; GPU: {torch.cuda.get_device_name(0)}")
PY
touch "$marker"

env_file="$PROJECT_DIR/dinov3_ucloud_env.sh"
{
  printf 'export DINOV3_ENV_DIR=%q\n' "$ENV_DIR"
  printf 'export PYTHON_BIN=%q\n' "$python_bin"
  printf 'export DINOV3_REPO=%q\n' "$DINOV3_REPO"
  printf 'export PROJECT_DIR=%q\n' "$PROJECT_DIR"
  printf 'export STUDY3_DIR=%q\n' "$PROJECT_DIR/Study3_SelfSupervised"
  printf 'export OUTPUT_ROOT=%q\n' "$OUTPUT_ROOT"
  if [[ -n "${DINOV3_IMAGE_ROOT:-}" ]]; then
    printf 'export IMAGE_ROOT=%q\n' "$DINOV3_IMAGE_ROOT"
  fi
  printf '%s\n' 'case ":$PATH:" in'
  printf '%s\n' '  *":$DINOV3_ENV_DIR/bin:"*) ;;'
  printf '%s\n' '  *) export PATH="$DINOV3_ENV_DIR/bin:$PATH" ;;'
  printf '%s\n' 'esac'
} > "$env_file"
echo "[Native704 Init] Ready. Shared outputs: $OUTPUT_ROOT"
printf '[Native704 Init] In the terminal: source %q\n' "$env_file"
