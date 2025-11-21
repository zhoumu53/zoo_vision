#!/bin/bash

# Elephant ReID Evaluation Script
# Usage: bash eval_elephant_resnet.sh <path_to_checkpoint>

# Set environment
export CUDA_VISIBLE_DEVICES=0

# Project paths
PROJECT_ROOT="/media/mu/zoo_vision/training/PoseGuidedReID"
CONFIG_FILE="${PROJECT_ROOT}/configs/elephant_resnet.yml"

# Check if checkpoint path provided
if [ -z "$1" ]; then
    echo "Usage: bash eval_elephant_resnet.sh <path_to_checkpoint>"
    echo "Example: bash eval_elephant_resnet.sh ./logs/elephant_resnet50/resnet50_120.pth"
    exit 1
fi

CHECKPOINT_PATH=$1

echo "======================================"
echo "Elephant ReID Evaluation - ResNet50"
echo "======================================"
echo "Config: ${CONFIG_FILE}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "======================================"

cd ${PROJECT_ROOT}

# Run evaluation
python tools/train.py \
    --config_file ${CONFIG_FILE} \
    --do_evaluation \
    TEST.WEIGHT ${CHECKPOINT_PATH}

echo "======================================"
echo "Evaluation completed!"
echo "======================================"
