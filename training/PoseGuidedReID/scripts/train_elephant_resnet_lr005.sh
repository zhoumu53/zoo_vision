#!/bin/bash

# Elephant ReID Training Script - ResNet50
# Usage: bash train_elephant_resnet.sh
cd ..

# Set environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT


# Training parameters
BATCH_SIZE=16
NUM_EPOCHS=120
LR=0.05

# Project paths
PROJECT_ROOT="/media/mu/zoo_vision/training/PoseGuidedReID"
CONFIG_FILE="${PROJECT_ROOT}/configs/elephant_resnet.yml"
EXP_NAME='lr001_bs16'
OUTPUT_DIR="${PROJECT_ROOT}/logs/elephant_resnet/${EXP_NAME}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "======================================"
echo "Elephant ReID Training - ResNet50"
echo "======================================"
echo "Config: ${CONFIG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Learning Rate: ${LR}"
echo "======================================"

cd ${PROJECT_ROOT}

# Run training
python3 tools/train.py \
    --config_file ${CONFIG_FILE} \
    --do_training \
    --do_inference \
    --notes "Elephant ReID with ResNet50 backbone, triplet + softmax loss" \
    SOLVER.IMS_PER_BATCH ${BATCH_SIZE} \
    SOLVER.MAX_EPOCHS ${NUM_EPOCHS} \
    SOLVER.BASE_LR ${LR} \
    SOLVER.STEPS "(20, 40, 70)" \
    SOLVER.OPTIMIZER_NAME 'SGD' \
    SOLVER.CHECKPOINT_PERIOD 10 \
    SOLVER.LOG_PERIOD 100 \
    SOLVER.WARMUP_EPOCHS 5 \
    MODEL.DIST_TRAIN False \
    MODEL.AGG_POSE_FEATURE False \
    DATALOADER.SAMPLER 'softmax' \
    OUTPUT_DIR ${OUTPUT_DIR} \
    TEST.MAX_RANK 20 \
    TEST.MAP_MAX_RANK True \
    


echo "======================================"
echo "Training completed!"
echo "Logs saved to: ${OUTPUT_DIR}"
echo "======================================"
