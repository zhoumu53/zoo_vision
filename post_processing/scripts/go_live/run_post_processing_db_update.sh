#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

source "$PROJECT_ROOT/env/bin/activate"

# Parse arguments
DATE=${1:-$(date -d "yesterday" +"%Y%m%d")}   ## default to yesterday's date (last night)
[[ $# -gt 0 ]] && shift

# ONLINE_CONFIG_FILE="$PROJECT_ROOT/data/config.json"
ONLINE_CONFIG_FILE='/home/dherrera/git/zoo_vision/data/config.json'
## LOAD RECORD ROOT FROM CONFIG FILE

# Validate config exists
[[ -f "$ONLINE_CONFIG_FILE" ]] || { echo "Config not found: $ONLINE_CONFIG_FILE" >&2; exit 2; }

# Read values (adjust JSON paths to your file)
RECORD_ROOT="$(jq -er '.record_root' "$ONLINE_CONFIG_FILE")"
OUTPUT_DIR="$(jq -er '.output_dir // (.record_root + "/demo")' "$ONLINE_CONFIG_FILE")"
# echo
echo "Record root: $RECORD_ROOT"
echo "Output dir: $OUTPUT_DIR"

# Setup logging
LOG_DIR="${RECORD_ROOT}/logs/post_processing/${DATE}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/log_$(date +"%Y%m%d_%H%M%S").log"
echo "Logging to: $LOG_FILE"

# Parse individual assignments (optional)  -- best performance if known individuals provided
cam1619_individuals="${1:-}"
cam1718_individuals="${2:-}"

if [[ $# -ge 2 ]]; then
  shift 2
elif [[ $# -eq 1 ]]; then
  shift 1
fi

######### SAVE TRACKS TO JSON ###########
python $PROJECT_ROOT/post_processing/tools/run_post_processing_full_night.py --date "$DATE" \
                                          --record-root "$RECORD_ROOT" \
                                          --output_dir "$OUTPUT_DIR" \
                                          --height 600 --width 1060 \
                                          --cam1619-individuals "$cam1619_individuals" \
                                          --cam1718-individuals "$cam1718_individuals" \
                                          --start_timestamp 16 \
                                          --end_timestamp 8 \
                                          --cross-camera-matching \
                                          --run-stitching &>> "$LOG_FILE"

echo "Feature extraction and stitching completed for date: $DATE"

##### ETHOGRAMS ###########
echo "Running ethogram analysis for date: $DATE"
python $PROJECT_ROOT/post_processing/analysis/activity_budget_analysis_per_day.py --date "$DATE" \
                                          --record_root "$RECORD_ROOT" &>> "$LOG_FILE"

echo "Ethogram analysis completed for date: $DATE"

##### UPDATE DB FROM TRACKS ###########
dates=("$DATE")

LOG_FILE="$LOG_DIR/db_log_at_$(date +"%Y%m%d_%H%M%S").log"
echo "Updating DB for dates: ${dates[*]} - logging to: $LOG_FILE"
python $PROJECT_ROOT/db/data_from_tracks.py --dir "$RECORD_ROOT"/tracks \
    --start_timestamp 16 \
    --end_timestamp 8 \
    --id_col 'identity_label' \
    --dates "${dates[@]}" &>> "$LOG_FILE"
echo "DB update completed for dates: ${dates[*]}"
