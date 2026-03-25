# read the csv file with empty videos, and delete the corresponding video files from the NAS
#!/usr/bin/env bash
set -euo pipefail

CSV_FILES_DIR="${1:-/media/ElephantsWD/empty_videos_to_be_deleted}"
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