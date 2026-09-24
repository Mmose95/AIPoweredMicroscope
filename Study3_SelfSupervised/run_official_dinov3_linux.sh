#!/usr/bin/env bash
# Portable WSL/UCloud launcher for the project-local manifest dataset.
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 DINOV3_REPO CONFIG MANIFEST IMAGE_ROOT OUTPUT_DIR GPUS [EXTRA_OVERRIDE ...]" >&2
  echo "Example GPUS: 1 on WSL, or the number allocated by UCloud." >&2
  exit 2
fi

dinov3_repo=$(realpath "$1")
config=$(realpath "$2")
manifest=$(realpath "$3")
image_root=$(realpath "$4")
output_dir=$5
gpus=$6
shift 6

[[ -f "$dinov3_repo/hubconf.py" ]] || { echo "Invalid DINOv3 repository: $dinov3_repo" >&2; exit 2; }
[[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
[[ -f "$manifest" ]] || { echo "Missing manifest: $manifest" >&2; exit 2; }
[[ -d "$image_root" ]] || { echo "Missing image root: $image_root" >&2; exit 2; }
[[ "$gpus" =~ ^[1-9][0-9]*$ ]] || { echo "GPUS must be a positive integer" >&2; exit 2; }

visible_gpus="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( gpus > visible_gpus )); then
  echo "Requested $gpus torchrun processes, but PyTorch sees only $visible_gpus CUDA devices." >&2
  echo "Use a GPUS value no greater than torch.cuda.device_count()." >&2
  exit 2
fi
echo "[DINOv3 launcher] Using $gpus of $visible_gpus visible CUDA devices"

mkdir -p "$output_dir"
export PYTHONPATH="$dinov3_repo${PYTHONPATH:+:$PYTHONPATH}"

resume_args=(--no-resume)
if [[ "${DINOV3_RESUME:-0}" == "1" ]]; then
  resume_args=()
fi

torchrun --standalone --nproc-per-node="$gpus" \
  "$(dirname "$0")/train_official_dinov3_manifest.py" \
  --config-file "$config" \
  --output-dir "$output_dir" \
  "${resume_args[@]}" \
  "train.dataset_path=MicroscopyManifest:manifest=$manifest:root=$image_root" \
  "$@"
