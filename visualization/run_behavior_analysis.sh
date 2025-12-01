#!/bin/bash

# Behavior Analysis Runner Script
# This script analyzes elephant behavior from stitched tracking results

# Input configuration
INPUT_JSONL="/media/dherrera/ElephantsWD/tracking_results/tracking_w_behavior_4cams/20250729/20250729_18/stitched_tracks/stitched_tracks.jsonl"
OUTPUT_DIR="/media/dherrera/ElephantsWD/tracking_results/tracking_w_behavior_4cams/20250729/20250729_18/behavior_analysis"

# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL="INFO"

# Run analysis
echo "========================================================================"
echo "ELEPHANT BEHAVIOR ANALYSIS"
echo "========================================================================"
echo "Input:  $INPUT_JSONL"
echo "Output: $OUTPUT_DIR"
echo ""

python3 behavior_analysis.py \
    --input "$INPUT_JSONL" \
    --output-dir "$OUTPUT_DIR" \
    --log-level "$LOG_LEVEL" \
    --invalid-zones-dir /media/mu/zoo_vision/data/invalid_zones \
    --filter-identity-jumps \
    --identity-jump-min-duration 10 \
    --identity-jump-window 120 \
    --bbox-iou-threshold 0.5

echo ""
echo "========================================================================"
echo "Analysis complete!"
echo "========================================================================"
echo ""
echo "Outputs:"
echo "  - behavior_summary.txt     : Human-readable report for zookeepers"
echo "  - timeline_data.csv        : Detailed data for further analysis"
echo "  - behavior_timeline.png    : Individual behavior timelines"
echo "  - room_occupancy.png       : Room occupancy over time"
echo "  - behavior_distribution.png: Behavior breakdown per elephant"
echo ""
