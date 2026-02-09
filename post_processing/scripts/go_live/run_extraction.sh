#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/env/bin/activate"

echo "Starting feature extraction and stitching process..."
echo "Project root: $PROJECT_ROOT"

#### 
NIGHT_CUTOFF=12
HOUR=$(date +%H)
if (( HOUR < NIGHT_CUTOFF )); then
  DATE=$(date -d "yesterday" +%Y%m%d)
else
  DATE=$(date +%Y%m%d)
fi
echo "Processing NIGHT=$DATE (Now: $(date))"

CAMERA_IDS=("$@")
if [[ ${#CAMERA_IDS[@]} -eq 0 ]]; then
  CAMERA_IDS=(016 017 018 019)
fi

# ONLINE_CONFIG_FILE="$PROJECT_ROOT/data/config.json"
ONLINE_CONFIG_FILE='/home/dherrera/git/zoo_vision/data/config.json'
## LOAD RECORD ROOT FROM CONFIG FILE

# Validate config exists
[[ -f "$ONLINE_CONFIG_FILE" ]] || { echo "Config not found: $ONLINE_CONFIG_FILE" >&2; exit 2; }

# Read values (adjust JSON paths to your file)
RECORD_ROOT="$(jq -er '.record_root' "$ONLINE_CONFIG_FILE")"
OUTPUT_DIR="$(jq -er '.output_dir // (.record_root + "/demo")' "$ONLINE_CONFIG_FILE")"
echo "Record root: $RECORD_ROOT"
echo "Output dir: $OUTPUT_DIR"

LOG_DIR="$RECORD_ROOT/logs/feature_extraction/${DATE}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/log_$(date +"%Y%m%d_%H%M%S").log"
echo "Logging to: $LOG_FILE"

for CAM_ID in "${CAMERA_IDS[@]}"; do
  echo "=== Extracting features for date: $DATE, camera: $CAM_ID ==="
  python3 $PROJECT_ROOT/post_processing/tools/extract_features_single_cam.py \
    --config $PROJECT_ROOT/post_processing/core/config/configs.yaml \
    --date "$DATE" \
    --record-root "$RECORD_ROOT" \
    --cam-id "$CAM_ID" \
    --run-night \
    --override processing.overwrite_reid=false \
               processing.overwrite_behavior=false &>> "$LOG_FILE"
done
echo "Feature extraction completed for date: $DATE, cameras: ${CAMERA_IDS[*]}"
