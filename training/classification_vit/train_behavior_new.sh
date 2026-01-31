
ROOT="/media/mu/zoo_vision/data/behaviour"
# TRAIN_ROOT="${ROOT}/sleep_v2"
EVAL_ROOT="${ROOT}/sleep_v4"

OUT_DIR="runs_sleep/two_stage_swinb_v2_to_v4_seed42"
MODEL="swin_base_patch4_window7_224"



  python run_sleep_swinb_two_heads.py \
    --train_root ${ROOT}/sleep_v1 \
                 ${ROOT}/sleep_v2 \
                ${ROOT}/sleep_v5 \
    --eval_root  ${EVAL_ROOT} \
    --out_dir    runs_sleep/two_heads_swinb_flip_cleandata \
    --model      swin_base_patch4_window7_224 \
    --hflip_swap_p 0.5 \
    --img_size   224 --batch_size 32 --epochs 100 \
    --lr  3e-5 --weight_decay 0.05 \
    --seed 42