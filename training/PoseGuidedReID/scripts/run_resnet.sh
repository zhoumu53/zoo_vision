cd ..


name=swin_bs64
batchsize=64

agg_pose_feature=False

config=./configs/resnet.yml
pretrained=./checkpoints/swin/swin_base_patch4_window7_224_22k.pth

notes='swinT'
model_name='_train'

##### MODIFY DONE #####

ANNO_ROOT=$DATA_ROOT/experiments/test_on_$year
IMG_DIR=$DATA_ROOT/data
log_dir=$DATA_ROOT/experiments/runs_6years_bs_${batchsize}
log_dir=$log_dir/$name/test_on_$year

mkdir -p $log_dir
### get current file name -> copy to the log dir
cp scripts/${BASH_SOURCE[0]} $log_dir
epoch='60'

python tools/train.py \
                    --do_train \
                    --do_inference \
                    --notes $notes \
                    --model_name $model_name \
                    --config_file $config \
                    SOLVER.MAX_EPOCHS $epochs \
                    DATASETS.ROOT_DIR $ANNO_ROOT \
                    DATASETS.IMG_DIR $IMG_DIR \
                    DATASETS.NAMES 'bear' \
                    OUTPUT_DIR $log_dir \
                    INPUT.PRE_SCALING $resize \
                    SOLVER.BASE_LR 0.1 \
                    SOLVER.STEPS "(20, 40, 70)" \
                    SOLVER.OPTIMIZER_NAME 'SGD' \
                    SOLVER.CHECKPOINT_PERIOD 10 \
                    SOLVER.IMS_PER_BATCH $batchsize \
                    SOLVER.LOG_PERIOD 40 \
                    SOLVER.WARMUP_EPOCHS 5 \
                    MODEL.PRETRAIN_PATH $pretrained \
                    MODEL.AGG_POSE_FEATURE $agg_pose_feature \
                    MODEL.DEVICE_ID "($gpu_id,)" \
                    MODEL.DIST_TRAIN False \
                    MODEL.POSE_WEIGHT $pose_weights \
                    DATALOADER.SAMPLER 'softmax' \
                    TEST.WEIGHT $test_weight \
                    TEST.FILTER_DATE True \
                    TEST.MAX_RANK 20 \
                    TEST.MAP_MAX_RANK True \

