#!/usr/bin/env bash
# UCloud initializer for the Native704 four-arm jobs. Reuses a prepared environment.
set -Eeuo pipefail

ENV_DIR="${DINOV3_ENV_DIR:-/work/CondaEnv/envs/phd-dinov3}"
PROJECT_DIR="${PROJECT_DIR:-/work/projects/myproj}"
DINOV3_REPO="${DINOV3_REPO:-/work/projects/dinov3}"
PROJECT_URL="${PROJECT_REPO_URL:-https://github.com/Mmose95/AIPoweredMicroscope.git}"
DINOV3_URL="${DINOV3_REPO_URL:-https://github.com/facebookresearch/dinov3.git}"
DINOV3_COMMIT="${DINOV3_COMMIT:-6876159a11b4df116f30f667f8c9888617df0751}"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "[Native704 Init][ERROR] Mount the prepared Conda environment at $ENV_DIR." >&2
  exit 1
fi

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

env_file="$PROJECT_DIR/dinov3_ucloud_env.sh"
{
  printf 'export DINOV3_ENV_DIR=%q\n' "$ENV_DIR"
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
