#!/usr/bin/env bash
set -Eeuo pipefail

# Minimal UCloud bootstrap. Clone/update the project before running any
# environment or storage detection so fixes in GitHub take effect immediately.
PROJECT_DIR="${PROJECT_DIR:-/work/projects/myproj}"
PROJECT_REPO_URL="${PROJECT_REPO_URL:-https://github.com/Mmose95/AIPoweredMicroscope.git}"

echo "[DINOv3 Bootstrap] Starting"
mkdir -p "$(dirname "$PROJECT_DIR")"

if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  echo "[DINOv3 Bootstrap] Cloning $PROJECT_REPO_URL into $PROJECT_DIR"
  git clone "$PROJECT_REPO_URL" "$PROJECT_DIR"
else
  echo "[DINOv3 Bootstrap] Updating $PROJECT_DIR"
  git -C "$PROJECT_DIR" pull --ff-only
fi

echo "[DINOv3 Bootstrap] Commit: $(git -C "$PROJECT_DIR" rev-parse --short HEAD)"
exec bash "$PROJECT_DIR/init_dinov3_ucloud.sh"

