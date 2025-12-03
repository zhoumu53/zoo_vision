cd ..

name=swin_base
batchsize=64

agg_pose_feature=False


config=./configs/swim_transformer/swin/swin_base_patch4_window7_224_22k.yaml
pretrained=./checkpoints/swin/swin_base_patch4_window7_224_22k.pth


CONFIG_FILE="${PROJECT_ROOT}/configs/elephant_swin.yml"
EXP_NAME='swin_lr01_bs64'
OUTPUT_DIR="${PROJECT_ROOT}/logs/elephant_resnet/${EXP_NAME}"

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
    --do_training \
    --do_inference \
    --notes "Elephant ReID with Swin Transformer, triplet + softmax loss" \
    SOLVER.IMS_PER_BATCH ${BATCH_SIZE} \
    SOLVER.MAX_EPOCHS ${NUM_EPOCHS} \
    SOLVER.BASE_LR 0.1 \
    SOLVER.STEPS "(20, 40, 70)" \
    SOLVER.OPTIMIZER_NAME 'SGD' \
    SOLVER.CHECKPOINT_PERIOD 10 \
    SOLVER.IMS_PER_BATCH $batchsize \
    SOLVER.LOG_PERIOD 40 \
    SOLVER.WARMUP_EPOCHS 5 \
    MODEL.DIST_TRAIN False \
    MODEL.AGG_POSE_FEATURE False \
    DATALOADER.SAMPLER 'softmax' \
    OUTPUT_DIR ${OUTPUT_DIR} \
    TEST.MAX_RANK 20 \
    TEST.MAP_MAX_RANK True \
    