#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ZOO_VISION_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

source "${ZOO_VISION_ROOT}/env/bin/activate"

export DATA_ROOT="${DATA_ROOT:-/mnt/camera_nas}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/media/ElephantsWD/empty_videos_to_be_deleted}"
mkdir -p "${OUTPUT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

cd "${PROJECT_ROOT}"
exec python3 -m empty_video_tool "$@"