#!/usr/bin/env bash
# Re-scan all camera folders on the NAS with the current model.
# Uses --rescan to ignore previous report.json history.
#
# Usage:
#   ./scripts/rescan_all.sh              # rescan all cameras
#   ./scripts/rescan_all.sh ZAG-ELP-CAM  # rescan only cameras matching pattern
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ZOO_VISION_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

source "${ZOO_VISION_ROOT}/env/bin/activate"

export DATA_ROOT="${DATA_ROOT:-/mnt/camera_nas}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/media/ElephantsWD/empty_videos_to_be_deleted}"

mkdir -p "${OUTPUT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

PATTERN="${1:-}"

cd "${PROJECT_ROOT}"

for CAMERA_DIR in "${DATA_ROOT}"/*/; do
    [ -d "${CAMERA_DIR}" ] || continue
    CAMERA_NAME="$(basename "${CAMERA_DIR}")"

    # Skip non-camera directories
    [[ "${CAMERA_NAME}" == @* ]] && continue
    [[ "${CAMERA_NAME}" == \#* ]] && continue

    # Filter by pattern if provided
    if [ -n "${PATTERN}" ] && [[ "${CAMERA_NAME}" != *"${PATTERN}"* ]]; then
        continue
    fi

    echo "========================================"
    echo "  Camera: ${CAMERA_NAME}"
    echo "========================================"
    python3 -u -m empty_video_tool.direct_scan "${CAMERA_NAME}" --rescan || \
        echo "ERROR: ${CAMERA_NAME} failed with exit code $?"
    echo
done

echo "=== All cameras scanned ==="
