#!/bin/bash

# Offline Track Stitching Runner
# Usage: bash run_offline_stitching.sh

set -e

# Activate the ReID virtual environment
REID_VENV="/media/mu/zoo_vision/venv"
if [ -d "$REID_VENV" ]; then
    source "$REID_VENV/bin/activate"
    echo "Activated virtual environment: $REID_VENV"
else
    echo "Warning: ReID venv not found at $REID_VENV"
fi

# Configuration
REID_CONFIG="/media/mu/zoo_vision/training/PoseGuidedReID/configs/elephant_resnet.yml"
REID_CHECKPOINT="/media/mu/zoo_vision/training/PoseGuidedReID/logs/elephant_resnet/lr001_bs16_softmax_triplet/net_best.pth"
DEVICE="cuda"

# Gallery database for identity anchoring (optional)
# Set to empty string to disable gallery matching
GALLERY="/media/mu/zoo_vision/training/PoseGuidedReID/logs/elephant_resnet/lr001_bs16_softmax_triplet/pred_features/train_iid/pytorch_result_e.npz"
GALLERY_DEVICE="cpu"
GALLERY_THRESHOLD=0.7
GALLERY_TOP_K=3

# Input directory containing tracking results
# This should contain subdirectories with *_tracks.jsonl files
# Example: /media/dherrera/ElephantsWD/tracking_results/tracking_w_behavior_4cams/20250729/20250729_18


date=$1
time=$2
if [ -z "$date" ] || [ -z "$time" ]; then
    echo "Usage: bash run_offline_stitching.sh <date> <time>"
    echo "Example: bash run_offline_stitching.sh 20250729 18"
    exit 1
fi

INPUT_DIR="/media/dherrera/ElephantsWD/tracking_results/tracking_w_behavior_4cams/$date/${date}_$time"

# Output directory (default: ${INPUT_DIR}/stitched_tracks)
# Leave empty to use default
OUTPUT_DIR=""

# Stage 1: Within-video stitching parameters
STAGE1_SIM_THRESHOLD=0.75    # ReID similarity threshold
STAGE1_TIME_GAP=0.1          # Max time gap (seconds)
STAGE1_SIZE_RATIO=0.4         # Min size ratio

# Stage 2: Cross-camera stitching parameters
STAGE2_SIM_THRESHOLD=0.85    # Stricter for cross-camera
STAGE2_TIME_GAP=0.1          # Shorter gap for cross-camera
STAGE2_SIZE_RATIO=0.6         # Stricter size matching

# Debug mode: limit frames processed (leave empty for full processing)
# Example: MAX_FRAMES=100 for quick testing
MAX_FRAMES=""

# Social group validation (requires gallery matching to be enabled)
# Set to "true" to enable, "" to disable
USE_SOCIAL_GROUPS="true"
# Action for conflicts: "report" (log only) or "remove" (remove lower-confidence matches)
SOCIAL_GROUP_ACTION="report"   

echo "==========================================="
echo "Offline Track Stitching Pipeline"
echo "==========================================="
echo ""
echo "Input directory: $INPUT_DIR"
if [ -z "$OUTPUT_DIR" ]; then
    echo "Output directory: ${INPUT_DIR}/stitched_tracks (default)"
else
    echo "Output directory: $OUTPUT_DIR"
fi
echo ""
echo "Stage 1 (within-video):"
echo "  - Similarity threshold: $STAGE1_SIM_THRESHOLD"
echo "  - Time gap: $STAGE1_TIME_GAP seconds"
echo "  - Size ratio: $STAGE1_SIZE_RATIO"
echo ""
echo "Stage 2 (cross-camera):"
echo "  - Similarity threshold: $STAGE2_SIM_THRESHOLD"
echo "  - Time gap: $STAGE2_TIME_GAP seconds"
echo "  - Size ratio: $STAGE2_SIZE_RATIO"
echo ""
if [ -n "$GALLERY" ]; then
    echo "Gallery matching:"
    echo "  - Gallery path: $GALLERY"
    echo "  - Threshold: $GALLERY_THRESHOLD"
    echo "  - Top-K: $GALLERY_TOP_K"
    echo ""
fi

# Build command
CMD="python3 offline_stitching.py \
    --input-dir \"$INPUT_DIR\" \
    --reid-config \"$REID_CONFIG\" \
    --reid-checkpoint \"$REID_CHECKPOINT\" \
    --device \"$DEVICE\" \
    --stage1-sim-threshold \"$STAGE1_SIM_THRESHOLD\" \
    --stage1-time-gap \"$STAGE1_TIME_GAP\" \
    --stage1-size-ratio \"$STAGE1_SIZE_RATIO\" \
    --stage2-sim-threshold \"$STAGE2_SIM_THRESHOLD\" \
    --stage2-time-gap \"$STAGE2_TIME_GAP\" \
    --stage2-size-ratio \"$STAGE2_SIZE_RATIO\" \
    --log-level INFO"

# Add output directory if specified
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""
fi

# Add gallery matching if specified
if [ -n "$GALLERY" ]; then
    CMD="$CMD --gallery \"$GALLERY\" \
        --gallery-device \"$GALLERY_DEVICE\" \
        --gallery-threshold \"$GALLERY_THRESHOLD\" \
        --gallery-top-k \"$GALLERY_TOP_K\""
fi

# Add max frames limit if specified (for debugging)
if [ -n "$MAX_FRAMES" ]; then
    CMD="$CMD --max-frames \"$MAX_FRAMES\""
fi

# Add social group validation if enabled
if [ "$USE_SOCIAL_GROUPS" = "true" ]; then
    CMD="$CMD --use-social-groups --social-group-action \"$SOCIAL_GROUP_ACTION\""
fi

# # Run offline stitching
eval $CMD

# Determine actual output directory for final message
if [ -z "$OUTPUT_DIR" ]; then
    ACTUAL_OUTPUT="${INPUT_DIR}/stitched_tracks"
else
    ACTUAL_OUTPUT="$OUTPUT_DIR"
fi

echo ""
echo "==========================================="
echo "Stitching complete!"
echo "==========================================="
echo "Output files:"
echo "  - Stitched JSONL: ${ACTUAL_OUTPUT}/stitched_tracks.jsonl"
echo "  - Summary stats: ${ACTUAL_OUTPUT}/stitching_summary.txt"
echo "  - Visualizations: ${ACTUAL_OUTPUT}/visualizations/"
echo ""
echo "Review the summary stats to see merge statistics."


# Input stitched tracks JSONL file
INPUT_JSONL="${ACTUAL_OUTPUT}/stitched_tracks.jsonl"

# Output directory for visualization frames
OUTPUT_DIR="${ACTUAL_OUTPUT}/visualizations_2x2"

FPS=2              # Target FPS (1.0 = 1 frame per second)
MAX_FRAMES=1000       # Maximum number of frames to generate

# Run visualization
python3 visualize_stitched_tracks.py \
    --input "$INPUT_JSONL" \
    --output-dir "$OUTPUT_DIR" \
    --fps "$FPS" \
    --max-frames "$MAX_FRAMES" \
    --log-level INFO 