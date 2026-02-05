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

ONLINE_CONFIG_FILE="$PROJECT_ROOT/data/config.json"
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
LOG_DIR="${RECORD_ROOT}/logs/post_processing"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/post_processing_night_${DATE}_log_at_$(date +"%Y%m%d_%H%M%S").log"

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
                                          --start_timestamp 18 \
                                          --end_timestamp 8 \
                                          --run-stitching &>> "$LOG_FILE"


##### UPDATE DB FROM TRACKS ###########
dates=("$DATE")
next_day=$(date -d "$DATE +1 day" +"%Y%m%d")
dates+=("$next_day")

echo "Updating DB for dates: ${dates[*]}"
python $PROJECT_ROOT/db/data_from_tracks.py --dir "$RECORD_ROOT"/tracks \
    --start_timestamp 18 \
    --end_timestamp 8 \
    --dates "${dates[@]}" &>> "$LOG_FILE"