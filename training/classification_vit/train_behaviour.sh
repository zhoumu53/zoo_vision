python training/classification_vit/train_classification.py \
    --model_name_or_path google/vit-base-patch16-224 \
    --train_dir /home/dherrera/data/elephants/behaviour/train \
    --validation_dir /home/dherrera/data/elephants/behaviour/val \
    --remove_unused_columns False \
    --dataloader_num_workers 12 \
    --dataloader_prefetch_factor 2 \
    --do_train \
    --do_eval \
    --eval_strategy epoch \
    --num_train_epochs 20 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --save_strategy epoch \
    --seed 1337 \
    --overwrite_output_dir \
    --ignore_mismatched_sizes \
    --no_flip \
    "$@"
    
