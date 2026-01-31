cd /media/mu/zoo_vision/post_processing/tools



# Parse arguments
DATE=${1:-$(date +"%Y%m%d")}
shift
CAMERA_IDS=("$@")
RECORD_ROOT="/media/mu/test_tracks"
OUTPUT_DIR="${RECORD_ROOT}/demo"

# Setup logging
LOG_DIR="${RECORD_ROOT}/logs/post_processing"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/post_processing_${DATE}_$(date +"%Y%m%d_%H%M%S").log"

CAMERA_IDS=(${CAMERA_IDS[@]:-016 017 018 019})
CAM_ID="${CAMERA_IDS[0]}"

#### LET IT RUN EVERY MORNING

# Parse individual assignments (optional)  -- best performance if known individuals provided
cam1619_individuals=${1:-""}
cam1718_individuals=${2:-""}
shift 2

DATE='20251115'  # test
#### TODO -- height, width here for mkv tracking video version --- filter out invalid tracks from another room
#### related to df-bbox-top2, bbox-left2, bbox-bottom2, bbox-right2 columns in tracks csv files
python run_post_processing_full_night.py --date "$DATE" \
                                          --record-root "$RECORD_ROOT" \
                                          --height 600 --width 1060 \
                                          --cam1619-individuals $cam1619_individuals \
                                          --cam1718-individuals $cam1718_individuals \
                                          --start_timestamp 18 \
                                          --end_timestamp 8 \
                                          --run-stitching \



# dates=("$DATE")
# next_day=$(date -d "$DATE +1 day" +"%Y%m%d")
# dates+=("$next_day")

# cd /media/mu/zoo_vision/db
# python data_from_tracks.py --dir "$RECORD_ROOT"/tracks \
#     --start_timestamp 18 \
#     --end_timestamp 8 \
#     --dates "${dates[@]}"