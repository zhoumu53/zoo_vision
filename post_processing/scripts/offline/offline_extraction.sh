#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"
source "$PROJECT_ROOT/env/bin/activate"

CAMERA_IDS=("$@")
if [[ ${#CAMERA_IDS[@]} -eq 0 ]]; then
  CAMERA_IDS=(016 017 018 019)
fi


RECORD_ROOT='/media/ElephantsWD/elephants/xmas'
OUTPUT_DIR="$RECORD_ROOT/demo"
echo "Record root: $RECORD_ROOT"
echo "Output dir: $OUTPUT_DIR"

### ls from record_root/tracks, check if start with '2025'

DATES=(
# 2025-02-09
# 2025-02-15
2025-02-28
2025-03-01
2025-03-15
2025-03-30
2025-03-31
2025-04-01
2025-04-15
2025-04-30
2025-05-01
2025-05-15
2025-05-30
2025-05-31
2025-06-01
2025-06-15
2025-06-30
2025-07-01
# 2025-07-15
# 2025-07-30
# 2025-08-15
# 2025-08-30
# 2025-08-31
# 2025-09-01
# 2025-09-15
# 2025-09-30
# 2025-10-01
# 2025-10-15
# 2025-10-30
# 2025-11-01
# 2025-11-15
# 2025-11-30
# 2025-12-01
# 2025-12-15
)

for DATE in "${DATES[@]}"; do
  echo "$DATE"
  skip_dates=("2025-11-15-" "2025-11-16-")
  ## if DATE in skip_dates, then skip
  if [[ " ${skip_dates[*]} " == *" $DATE "* ]]; then
    echo "=== Skipping date: $DATE ==="
    continue
  fi

  LOG_DIR="$RECORD_ROOT/logs/feature_extraction/${DATE}"
  mkdir -p "$LOG_DIR"

  for CAM_ID in "${CAMERA_IDS[@]}"; do
    LOG_FILE="$LOG_DIR/log_cam${CAM_ID}_$(date +"%Y%m%d_%H%M%S").log"
    echo "=== Extracting features for date: $DATE, camera: $CAM_ID ==="
    echo "Logging to: $LOG_FILE"
    python3 $PROJECT_ROOT/post_processing/tools/offline_extract_features_single_cam.py \
      --config $PROJECT_ROOT/post_processing/core/config/configs.yaml \
      --date "$DATE" \
      --record-root "$RECORD_ROOT" \
      --cam-id "$CAM_ID" \
      --run-night \
      --override processing.overwrite_reid=true \
                 processing.overwrite_behavior=true &> "$LOG_FILE"
  done

done