set -e

DEVICE=8
TOTAL_BS=2048
DEVICE_BS=64
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BS / DEVICE_BS / DEVICE))
YAML_CONFIG="c.yaml"

BASE_MODEL="bert-vocab-sb-tn"
BASE_NAME=$BASE_MODEL
SUFFIX="VT-emb-b5"
DO_TRAIN_STEP_0=false

if [ "$DO_TRAIN_STEP_0" = true ]; then
    sed -i -E "s|^model_name_or_path:.*|model_name_or_path: ${BASE_MODEL}|" "$YAML_CONFIG"
    sed -i -E "s|^tokenizer_name:.*|tokenizer_name: ${BASE_MODEL}|" "$YAML_CONFIG"
    sed -i -E "s|^output_dir:.*|output_dir: output/paper/bi/$BASE_NAME-$SUFFIX-0/final|" "$YAML_CONFIG"
    bash run_train_eval.sh $YAML_CONFIG
fi

STEPS=(20000 10000 5000)
for STEP in "${STEPS[@]}"
do
    torchrun --nproc_per_node=$DEVICE --master_port 29501 run_mlm.py \
        --model_name_or_path $BASE_MODEL \
        --train_file 'data/wikibook.ml128.jsonl' \
        --tokenizer_name bert-base-uncased \
        --max_seq_length 128 \
        --mlm_probability 0.3 \
        --per_device_train_batch_size $DEVICE_BS \
        --per_device_eval_batch_size $DEVICE_BS \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --do_train \
        --output_dir pretrain/$BASE_NAME-$SUFFIX-$STEP \
        --dataloader_drop_last \
        --dataloader_num_workers 8 \
        --logging_steps 50 \
        --max_steps $STEP \
        --save_steps $STEP \
        --warmup_steps 4000 \
        --optim adamw_torch \
        --report_to tensorboard \
        --lr_scheduler_type cosine \
        --learning_rate 3e-4 \
        --weight_decay 0.01 \
        --overwrite_output_dir \
        --fp16 \
        --additional_tokens "additional_tokens_v2.json" \
        --log_level info \
        --train_only_embeddings

    CKPT_PATH="pretrain/$BASE_NAME-$SUFFIX-$STEP/checkpoint-$STEP"
    OUT_DIR="output/paper/bi/$BASE_NAME-$SUFFIX-$STEP/final"

    git restore $YAML_CONFIG
    sed -i -E "s|^model_name_or_path:.*|model_name_or_path: ${CKPT_PATH}|" "$YAML_CONFIG"
    sed -i -E "s|^tokenizer_name:.*|tokenizer_name: ${CKPT_PATH}|" "$YAML_CONFIG"
    sed -i -E "s|^output_dir:.*|output_dir: ${OUT_DIR}|" "$YAML_CONFIG"

    bash run_train_eval.sh $YAML_CONFIG
done