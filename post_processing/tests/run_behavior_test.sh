#!/bin/bash
# Test script for BehaviorInference

# Example usage:
# 1. Single image test:
#    bash run_behavior_test.sh --image /path/to/image.jpg
#
# 2. Multiple images test:
#    bash run_behavior_test.sh --images /path/to/img1.jpg /path/to/img2.jpg /path/to/img3.jpg
#
# 3. Video test:
#    bash run_behavior_test.sh --video /path/to/video.mp4
#
# 4. Video with sampling:
#    bash run_behavior_test.sh --video /path/to/video.mp4 --n-frames 20
#
# 5. Video with sampling comparison:
#    bash run_behavior_test.sh --video /path/to/video.mp4 --test-sampling

PROJECT_ROOT= ""
# Default model path (update this to your model location)
MODEL_PATH="/media/mu/zoo_vision/models/sleep/vit/v2_no_validation/config.ptc"

# Activate environment if needed
if [ -d "/media/mu/zoo_vision/env" ]; then
    source /media/mu/zoo_vision/env/bin/activate
fi

# Run test script
python3 /media/mu/zoo_vision/post_processing/tests/test_behavior_inference.py \
    --model "$MODEL_PATH" \
    --image "/media/mu/zoo_vision/data/behaviour/sleep_v3/02_sleeping_left/zag_elp_cam_016_004935.jpg"\
