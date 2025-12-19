# todo: save additional tokens

set -e

cd ..

DEVICE=8
TOTAL_BS=512
DEVICE_BS=64
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BS / DEVICE_BS / DEVICE))

BASE_MODEL="source_model/bert-base-uncased-dolma-chem-only-query-generated"
OUTPUT_DIR="mlm/bert-base-uncased-dolma-chem-only-query-generated"
TRAIN_FILE="chem/dolma-chem-only-query-generated.json"

torchrun --nproc_per_node=$DEVICE --master_port 29502 run_mlm.py \
    --model_name_or_path chem/$BASE_MODEL \
    --train_file $TRAIN_FILE \
    --tokenizer_name chem/$BASE_MODEL \
    --max_seq_length 512 \
    --mlm_probability 0.3 \
    --per_device_train_batch_size $DEVICE_BS \
    --per_device_eval_batch_size $DEVICE_BS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --do_train \
    --output_dir chem/$OUTPUT_DIR \
    --dataloader_drop_last \
    --dataloader_num_workers 8 \
    --logging_steps 50 \
    --max_steps 5000 \
    --save_steps 1000 \
    --warmup_ratio 0.1 \
    --optim adamw_torch \
    --report_to tensorboard \
    --lr_scheduler_type linear \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --overwrite_output_dir \
    --fp16 \
    --additional_tokens chem/$BASE_MODEL/additional_tokens.json \
    --log_level info

cd chem

BASE_MODEL="$OUTPUT_DIR/checkpoint-5000"

OUTPUT_DIR="ft/bert-base-uncased-dolma-chem-only-query-generated"

torchrun --nproc_per_node=8 --master_port 29501 train_chem.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --train_dataset "BASF-AI/dolma-chem-only-query-generated"

CUDA_VISIBLE_DEVICES=0 python eval_chem.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --max_active_dims 128 &