#!/bin/bash

# Visualize Stitched Tracks in 2x2 Multi-Camera Grid
# Usage: bash run_visualize_stitched.sh

set -e

# Input stitched tracks JSONL file
INPUT_JSONL="/media/dherrera/ElephantsWD/tracking_results/tracking_w_behavior_4cams/20250729/20250729_18/stitched_tracks/stitched_tracks.jsonl"

# Output directory for visualization frames
OUTPUT_DIR="/media/dherrera/ElephantsWD/tracking_results/tracking_w_behavior_4cams/20250729/20250729_18/stitched_tracks/visualizations_2x2"

# Visualization parameters
FPS=1.0              # Target FPS (1.0 = 1 frame per second)
MAX_FRAMES=500       # Maximum number of frames to generate

echo "==========================================="
echo "Stitched Tracks Multi-Camera Visualization"
echo "==========================================="
echo ""
echo "Input: $INPUT_JSONL"
echo "Output: $OUTPUT_DIR"
echo "FPS: $FPS"
echo "Max frames: $MAX_FRAMES"
echo ""

# Run visualization
python3 visualize_stitched_tracks.py \
    --input "$INPUT_JSONL" \
    --output-dir "$OUTPUT_DIR" \
    --fps "$FPS" \
    --max-frames "$MAX_FRAMES" \
    --log-level INFO

echo ""
echo "==========================================="
echo "Visualization complete!"
echo "==========================================="
echo "Frames saved to: $OUTPUT_DIR"
echo ""
echo "To create a video from frames:"
echo "ffmpeg -framerate $FPS -pattern_type glob -i '$OUTPUT_DIR/frame_*.jpg' -c:v libx264 -pix_fmt yuv420p $OUTPUT_DIR/stitched_tracks_2x2.mp4"
