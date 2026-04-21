#!/usr/bin/env bash

#### 0 18 * * * /media/mu/zoo_vision/clean_nas_empty_videos/scripts/cron_daily_scan.sh
# This script is intended to be run as a daily cron job to scan for empty videos from camera 16-19 in the NAS.
# It will log the output of each scan to a daily log file in the logs directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ZOO_VISION_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

source "${ZOO_VISION_ROOT}/env/bin/activate"

export DATA_ROOT="${DATA_ROOT:-/mnt/camera_nas}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/media/ElephantsWD/empty_videos_to_be_deleted}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "${OUTPUT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/daily_scan_$(date +%Y%m%d).log"

echo "=== Daily scan started at $(date) ===" >> "${LOG_FILE}"

for FOLDER in "${DATA_ROOT}"/ZAG-ELP-CAM-01*/; do
    [ -d "${FOLDER}" ] || continue
    FOLDER_NAME="$(basename "${FOLDER}")"
    echo "--- Scanning ${FOLDER_NAME} at $(date) ---" >> "${LOG_FILE}"
    python3 -m empty_video_tool.direct_scan "${FOLDER_NAME}" >> "${LOG_FILE}" 2>&1 || \
        echo "ERROR: ${FOLDER_NAME} failed with exit code $?" >> "${LOG_FILE}"
done

echo "=== Daily scan finished at $(date) ===" >> "${LOG_FILE}"
