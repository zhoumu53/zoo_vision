#!/bin/bash

# Elephant ReID Training Script - ResNet50
# Usage: bash train_elephant_resnet.sh
cd ..

# Set environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT


# Training parameters
BATCH_SIZE=128
NUM_EPOCHS=100
LR=0.0003

# Project paths
PROJECT_ROOT="/media/mu/zoo_vision/training/PoseGuidedReID"
CONFIG_FILE="${PROJECT_ROOT}/configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml"
EXP_NAME="swin_adamw_lr0003_bs${BATCH_SIZE}_softmax_triplet_Fulldata"
OUTPUT_DIR="/media/ElephantsWD/elephants/reid_models/${EXP_NAME}"

pretrained=./checkpoints/swin_base_patch4_window7_224_22k.pth

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "======================================"
echo "Elephant ReID Training - Swin Transformer"
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
    --do_inference \
    --notes "Elephant ReID with Swin Transformer backbone, triplet + softmax loss" \
    DATASETS.ROOT_DIR '/media/mu/zoo_vision/data/reid_time_split' \
    DATASETS.IMG_DIR '/media/mu/zoo_vision/data/reid_time_split' \
    DATASETS.NAMES 'elephant' \
    SOLVER.IMS_PER_BATCH ${BATCH_SIZE} \
    SOLVER.MAX_EPOCHS ${NUM_EPOCHS} \
    SOLVER.BASE_LR ${LR} \
    SOLVER.STEPS "(40, 70)" \
    SOLVER.OPTIMIZER_NAME 'AdamW' \
    SOLVER.WARMUP_EPOCHS 5 \
    SOLVER.WARMUP_FACTOR 0.01 \
    SOLVER.WARMUP_METHOD 'linear' \
    SOLVER.WEIGHT_DECAY 0.0001 \
    SOLVER.WEIGHT_DECAY_BIAS 0.0001 \
    SOLVER.CHECKPOINT_PERIOD 10 \
    SOLVER.LOG_PERIOD 100 \
    MODEL.DIST_TRAIN False \
    MODEL.RESUME True \
    MODEL.AGG_POSE_FEATURE False \
    MODEL.PRETRAIN_PATH ${pretrained} \
    DATALOADER.SAMPLER 'softmax_triplet' \
    DATALOADER.NUM_INSTANCE 8 \
    OUTPUT_DIR ${OUTPUT_DIR} \
    TEST.MAX_RANK 20 \
    TEST.MAP_MAX_RANK True \


# python3 tools/predict.py \
#     --config_file ${CONFIG_FILE} \
#     DATASETS.TEST_ROOT_DIR '/media/mu/zoo_vision/data/reid_time_split/val' \
#     DATASETS.TEST_IMG_DIR '/media/mu/zoo_vision/data/reid_time_split/val' \
#     DATASETS.NAMES 'base' \

# echo "======================================"
# echo "Training completed!"
# echo "Logs saved to: ${OUTPUT_DIR}"
# echo "======================================"
