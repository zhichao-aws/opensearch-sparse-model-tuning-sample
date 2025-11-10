set -e

DEVICE=8
TOTAL_BS=2048
DEVICE_BS=256
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BS / DEVICE_BS / DEVICE))

# torchrun --nproc_per_node=$DEVICE --master_port 29501 run_sap.py \
#     --model_name_or_path alignment-modernbert-base \
#     --per_device_train_batch_size $DEVICE_BS \
#     --per_device_eval_batch_size $DEVICE_BS \
#     --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#     --do_train \
#     --output_dir output_sap/run_freeze_140k \
#     --dataloader_drop_last \
#     --logging_steps 50 \
#     --max_steps 140000 \
#     --save_steps 50000 \
#     --warmup_steps 4000 \
#     --optim adamw_torch \
#     --lr_scheduler_type cosine \
#     --learning_rate 3e-5 \
#     --weight_decay 0.01 \
#     --overwrite_output_dir \
#     --train_only_embeddings \
#     --fp16

# torchrun --nproc_per_node=$DEVICE --master_port 29501 run_sap.py \
#     --model_name_or_path alignment-modernbert-base \
#     --per_device_train_batch_size $DEVICE_BS \
#     --per_device_eval_batch_size $DEVICE_BS \
#     --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#     --do_train \
#     --output_dir output_sap/run_140k \
#     --dataloader_drop_last \
#     --logging_steps 50 \
#     --max_steps 140000 \
#     --save_steps 50000 \
#     --warmup_steps 4000 \
#     --optim adamw_torch \
#     --lr_scheduler_type cosine \
#     --learning_rate 3e-5 \
#     --weight_decay 0.01 \
#     --overwrite_output_dir \
#     --fp16


# torchrun --nproc_per_node=$DEVICE --master_port 29501 run_sap.py \
#     --model_name_or_path alignment-modernbert-base \
#     --per_device_train_batch_size $DEVICE_BS \
#     --per_device_eval_batch_size $DEVICE_BS \
#     --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#     --do_train \
#     --output_dir output_sap/run_freeze_40k \
#     --dataloader_drop_last \
#     --logging_steps 50 \
#     --max_steps 40000 \
#     --save_steps 50000 \
#     --warmup_steps 4000 \
#     --optim adamw_torch \
#     --lr_scheduler_type cosine \
#     --learning_rate 3e-5 \
#     --weight_decay 0.01 \
#     --overwrite_output_dir \
#     --train_only_embeddings \
#     --fp16

# torchrun --nproc_per_node=$DEVICE --master_port 29501 run_sap.py \
#     --model_name_or_path alignment-modernbert-base \
#     --per_device_train_batch_size $DEVICE_BS \
#     --per_device_eval_batch_size $DEVICE_BS \
#     --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
#     --do_train \
#     --output_dir output_sap/run_40k \
#     --dataloader_drop_last \
#     --logging_steps 50 \
#     --max_steps 40000 \
#     --save_steps 50000 \
#     --warmup_steps 4000 \
#     --optim adamw_torch \
#     --lr_scheduler_type cosine \
#     --learning_rate 3e-5 \
#     --weight_decay 0.01 \
#     --overwrite_output_dir \
#     --fp16