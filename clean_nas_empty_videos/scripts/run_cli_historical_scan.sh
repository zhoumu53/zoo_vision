#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export HOST_VIDEO_ROOT="${HOST_VIDEO_ROOT:-/mnt/camera_nas}"
export HOST_MODEL_ROOT="${HOST_MODEL_ROOT:-${PROJECT_ROOT}/models}"
export HOST_EMPTY_EXPORT_ROOT="${HOST_EMPTY_EXPORT_ROOT:-/media/ElephantsWD/empty_videos_to_be_deleted}"
export TOP_LEVEL_FOLDER_GLOB="${TOP_LEVEL_FOLDER_GLOB:-ELP-Kamera-*}"

mkdir -p "${PROJECT_ROOT}/runs" "${PROJECT_ROOT}/models" "${HOST_EMPTY_EXPORT_ROOT}"

TTY_FLAGS=()
if [[ ! -t 0 || ! -t 1 ]]; then
  TTY_FLAGS+=("-T")
fi

cd "${PROJECT_ROOT}"
exec docker compose run --rm --build "${TTY_FLAGS[@]}" app empty-video-cli "$@"
