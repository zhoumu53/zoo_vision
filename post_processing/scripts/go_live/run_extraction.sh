#!/usr/bin/env bash
set -euo pipefail

cd ../../..   #### zoo_vision root

DATE="${1:-$(date -d "yesterday" +%Y%m%d)}"
[[ $# -gt 0 ]] && shift

CAMERA_IDS=("$@")
if [[ ${#CAMERA_IDS[@]} -eq 0 ]]; then
  CAMERA_IDS=(016 017 018 019)
fi

ONLINE_CONFIG_FILE='data/config.json'
## LOAD RECORD ROOT FROM CONFIG FILE

# Validate config exists
[[ -f "$ONLINE_CONFIG_FILE" ]] || { echo "Config not found: $ONLINE_CONFIG_FILE" >&2; exit 2; }

# Read values (adjust JSON paths to your file)
RECORD_ROOT="$(jq -er '.record_root' "$ONLINE_CONFIG_FILE")"
OUTPUT_DIR="$(jq -er '.output_dir // (.record_root + "/demo")' "$ONLINE_CONFIG_FILE")"
# echo
echo "Record root: $RECORD_ROOT"
echo "Output dir: $OUTPUT_DIR"

LOG_DIR="$RECORD_ROOT/logs/feature_extraction"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/extraction_night_${DATE}_log_at_$(date +"%Y%m%d_%H%M%S").log"

for CAM_ID in "${CAMERA_IDS[@]}"; do
  echo "=== Extracting features for date: $DATE, camera: $CAM_ID ==="
  python3 post_processing/tools/extract_features_single_cam.py --config post_processing/core/config/configs.yaml \
    --date "$DATE" \
    --record-root "$RECORD_ROOT" \
    --cam-id "$CAM_ID" \
    --run-night \
    --override processing.overwrite_reid=false \
               processing.overwrite_behavior=false >> "$LOG_FILE"
done