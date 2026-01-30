
cd /media/mu/zoo_vision/post_processing/tools 


# Parse arguments
DATE=${1:-$(date +"%Y%m%d")}   ## default to today
shift
CAMERA_IDS=("$@")
RECORD_ROOT="/media/ElephantsWD/elephants/long-term-data"
RECORD_ROOT='/media/mu/test_tracks'
OUTPUT_DIR="${RECORD_ROOT}/demo"

# Setup logging
LOG_DIR=$RECORD_ROOT/logs/test
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/extraction_${DATE}_$(date +"%Y%m%d_%H%M%S").log"

CAMERA_IDS=(${CAMERA_IDS[@]:-016 017 018 019})
CAM_ID="${CAMERA_IDS[0]}"

##### overwrite_reid, overwrite_behavior= False: only process new tracks #####

SAMPLE_RATE=1 #### SAVE EVERY FRAME FOR GOOD REID QUALITY -
DATE='20251115'

for CAM_ID in "${CAMERA_IDS[@]}"; 
do
    for date in "$DATE";
    do
        python3 extract_features_single_cam.py --config ../core/config/configs.yaml \
                                                --date "$date" \
                                                --record-root "$RECORD_ROOT" \
                                                --cam-id "$CAM_ID" \
                                                --override processing.overwrite_reid=true \
                                                        processing.overwrite_behavior=true \

    done
done
