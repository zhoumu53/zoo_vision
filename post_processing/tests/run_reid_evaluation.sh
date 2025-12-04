#!/bin/bash

# Example script to run ReID evaluation test
# Adjust paths according to your setup

IMAGE_DIR="/media/mu/zoo_vision/data/reid_time_split/val"
GALLERY_IMAGE_DIR="/media/mu/zoo_vision/data/reid_time_split/train"
NUM_CLASSES=5

CONFIG_FILE='/media/mu/zoo_vision/training/PoseGuidedReID/configs/elephant_resnet.yml'
MODEL='lr001_bs16_softmax_triplet'
RESULT_DIR="/media/mu/zoo_vision/training/PoseGuidedReID/logs/elephant/${MODEL}"
CHECKPOINT="${RESULT_DIR}/net_best.pth"

# IMAGE_DIR="${RESULT_DIR}/pred_features/val_iid/pytorch_result_e.npz"
# GALLERY_IMAGE_DIR="${RESULT_DIR}/pred_features/train_iid/pytorch_result_e.npz"


# CONFIG_FILE="/media/mu/zoo_vision/training/PoseGuidedReID/configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml"
# CHECKPOINT="/media/dherrera/ElephantsWD/reid_models/logs/swin_adamw_lr0003_bs64_softmax_triplet/net_best.pth"

# IMAGE_DIR="/media/dherrera/ElephantsWD/reid_models/logs/swin_adamw_lr0003_bs64_softmax_triplet/pred_features/val_iid/pytorch_result_e.npz"
# GALLERY_IMAGE_DIR="/media/dherrera/ElephantsWD/reid_models/logs/swin_adamw_lr0003_bs64_softmax_triplet/pred_features/train_iid/pytorch_result_e.npz"

# Run the evaluation script
python /media/mu/zoo_vision/post_processing/tests/test_reid_evaluation.py \
    --config "$CONFIG_FILE" \
    --checkpoint "$CHECKPOINT" \
    --image_dir "$IMAGE_DIR" \
    --gallery_image_dir "$GALLERY_IMAGE_DIR" \
    --num_classes $NUM_CLASSES \
    --device cuda \
    --batch_size 32 \
    --max_rank 20 \
    --verbose
