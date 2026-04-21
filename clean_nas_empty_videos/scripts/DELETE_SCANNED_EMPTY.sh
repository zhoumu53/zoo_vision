#!/usr/bin/env bash
# Read CSV files with empty videos and delete the corresponding video files from the NAS.
# CSVs are located under OUTPUT_ROOT/{camera}/{date}/{date}.csv
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CSV_FILES_DIR="${1:-${OUTPUT_ROOT:-/media/ElephantsWD/empty_videos_to_be_deleted}}"
if [[ ! -d "${CSV_FILES_DIR}" ]]; then
  echo "Directory ${CSV_FILES_DIR} does not exist"
  exit 1
fi

find "${CSV_FILES_DIR}" -name "*.csv" | while read -r csv_file; do
    echo "Processing ${csv_file}..."

    ## read items from csv file, skipping the header
    tail -n +2 "${csv_file}" | while IFS=, read -r video_path _; do
        if [[ -f "${video_path}" ]]; then
        echo "Deleting ${video_path}..."
        # rm "${video_path}"              ##################################### UNCOMMENT THIS TO ACTUALLY DELETE FILES
        else
        echo "File ${video_path} does not exist, skipping."
        fi
    done
done