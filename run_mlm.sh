DEVICE=8
TOTAL_BS=2048
DEVICE_BS=64
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BS / DEVICE_BS / DEVICE))

torchrun --nproc_per_node=$DEVICE --master_port 29501 run_mlm.py \
    --model_name_or_path modernbert-with-gte-vocab \
    --train_file data/wikibook.ml128.jsonl \
    --max_seq_length 128 \
    --mlm_probability 0.3 \
    --per_device_train_batch_size $DEVICE_BS \
    --per_device_eval_batch_size $DEVICE_BS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --do_train \
    --output_dir pretrain/mdbert-mlm+msmarco \
    --dataloader_drop_last \
    --dataloader_num_workers 8 \
    --logging_steps 50 \
    --max_steps 140000 \
    --save_steps 20000 \
    --warmup_steps 4000 \
    --optim adamw_torch \
    --report_to tensorboard \
    --lr_scheduler_type cosine \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --overwrite_output_dir \
    --fp16