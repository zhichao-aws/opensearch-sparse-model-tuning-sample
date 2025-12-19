set -e

YAML_CONFIG="c.yaml"

STEPS=(500 1000 2000 3000 5000 10000 15000)
for STEP in "${STEPS[@]}"
do  
    BASE_MODEL="pretrain/bert-vocab-sb-tn-VT-b5-$STEP/checkpoint-$STEP"
    TARGET_MODEL="bert-vocab-sb-tn-VT-$STEP-P60"
    python probe_activation_percentiles.py \
        --model_id $BASE_MODEL \
        --cut_percent 60 \
        --save_path pretrain/$TARGET_MODEL

    git restore $YAML_CONFIG
    sed -i -E "s|^model_name_or_path:.*|model_name_or_path: pretrain/$TARGET_MODEL|" "$YAML_CONFIG"
    sed -i -E "s|^tokenizer_name:.*|tokenizer_name: pretrain/$TARGET_MODEL|" "$YAML_CONFIG"
    sed -i -E "s|^output_dir:.*|output_dir: output/paper/bi/$TARGET_MODEL/final|" "$YAML_CONFIG"

    bash run_train_eval.sh $YAML_CONFIG
done