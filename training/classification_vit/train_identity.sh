python training/classification_vit/train_classification.py \
    --model_name_or_path /home/dherrera/git/zoo_vision/models/identity/vit/all_freeze_vit \
    --train_dir /home/dherrera/data/elephants/identity/dataset/v1/train_curated \
                /home/dherrera/data/elephants/identity/dataset/v1/val \
                /home/dherrera/data/elephants/identity/dataset/d2/train \
                /home/dherrera/data/elephants/identity/dataset/certainty/train/good \
                /home/dherrera/data/elephants/identity/dataset/certainty/val \
                /home/dherrera/data/elephants/identity/dataset/id3 \
                /home/dherrera/data/elephants/identity/dataset/v4 \
    --train_val_split 0.2 \
    --freeze_layers embeddings \
    --remove_unused_columns False \
    --dataloader_num_workers 12 \
    --dataloader_prefetch_factor 2 \
    --do_train \
    --do_eval \
    --eval_on_start true \
    --eval_strategy epoch \
    --num_train_epochs 200 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --load_best_model_at_end True \
    --save_total_limit 20 \
    --metric_for_best_model loss \
    --save_strategy epoch \
    --seed 1337 \
    --overwrite_output_dir \
    --ignore_mismatched_sizes \
    "$@"
    
