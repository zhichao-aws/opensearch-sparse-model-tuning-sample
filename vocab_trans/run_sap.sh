set -e

DEVICE=8
TOTAL_BS=2048
DEVICE_BS=32
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BS / DEVICE_BS / DEVICE))

torchrun --nproc_per_node=$DEVICE --master_port 29503 run_sap.py \
    --model_name_or_path alignment-modernbert-base \
    --per_device_train_batch_size $DEVICE_BS \
    --per_device_eval_batch_size $DEVICE_BS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --do_train \
    --output_dir output_sap/20k \
    --dataloader_drop_last \
    --logging_steps 50 \
    --max_steps 20000 \
    --save_steps 10000 \
    --warmup_steps 4000 \
    --optim adamw_torch \
    --lr_scheduler_type cosine \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --overwrite_output_dir \
    --fp16