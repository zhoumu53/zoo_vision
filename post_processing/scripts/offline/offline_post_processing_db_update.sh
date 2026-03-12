#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_ROOT"
echo "Project root: $PROJECT_ROOT"

source "$PROJECT_ROOT/env/bin/activate"
export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT" 

RECORD_ROOT='/media/ElephantsWD/elephants/xmas'
RECORD_ROOT='/media/ElephantsWD/elephants/test_dan/results'
OUTPUT_DIR="$RECORD_ROOT/demo"
echo "Record root: $RECORD_ROOT"
echo "Output dir: $OUTPUT_DIR"


DATES=(
# 2025-02-09
# 2025-02-15
# 2025-02-28
# 2025-03-01
# 2025-03-15
# 2025-03-30
# 2025-03-31
# 2025-04-01
# 2025-04-15
# 2025-04-30
# 2025-05-01
# 2025-05-15
# 2025-05-30
# 2025-05-31
# 2025-06-01
# 2025-06-15
# 2025-06-30
# 2025-07-01
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
2025-11-01
2025-11-15
2025-11-30
2025-12-01
2025-12-15
)

failures=0

# DATES=(
#   2026-03-07
# )
echo "Found dates: ${DATES[*]}"

for DATE in "${DATES[@]}"; do
  LOG_DIR="${RECORD_ROOT}/logs/post_processing/${DATE}"
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/log_$(date +"%Y%m%d_%H%M%S").log"
  echo "Logging to: $LOG_FILE"

  echo "=== [$DATE] Stage 1: offline_post_processing_full_night.py ===" | tee -a "$LOG_FILE"

  set +e
  python "$PROJECT_ROOT/post_processing/tools/offline_post_processing_full_night.py" \
      --date "$DATE" \
      --record-root "$RECORD_ROOT" \
      --output_dir "$OUTPUT_DIR" \
      --height 600 --width 1060 \
      --start_timestamp 18 \
      --end_timestamp 8 \
      --cross-camera-matching \
      --run-stitching #&>> "$LOG_FILE"
  rc1=$?
  set -e

  if [[ $rc1 -ne 0 ]]; then
    echo "[$DATE] Stage 1 FAILED (exit=$rc1). Skipping DB update and continuing." | tee -a "$LOG_FILE"
    ((failures++))
    continue
  fi

  echo "[$DATE] Stage 1 OK. Stage 2: DB update" | tee -a "$LOG_FILE"


  echo "Running ethogram analysis for date: $DATE"
  python $PROJECT_ROOT/post_processing/analysis/activity_budget_analysis_per_day.py --date "$DATE" \
                                            --record_root "$RECORD_ROOT" #&>> "$LOG_FILE"

  echo "Ethogram analysis completed for date: $DATE"

  set +e
  python "$PROJECT_ROOT/db/data_from_tracks.py" \
      --dir "$RECORD_ROOT/tracks" \
      --start_timestamp 16 \
      --end_timestamp 8 \
      --delete-existing-night \
      --id_col "identity_label" \
      --dates "$DATE" #&>> "$LOG_FILE"
  rc2=$?
  set -e

  if [[ $rc2 -ne 0 ]]; then
    echo "[$DATE] Stage 2 FAILED (exit=$rc2). Continuing." | tee -a "$LOG_FILE"
    ((failures++))
    continue
  fi

  echo "[$DATE] DONE" | tee -a "$LOG_FILE"
done

echo "All dates processed. Failures: $failures"
exit 0