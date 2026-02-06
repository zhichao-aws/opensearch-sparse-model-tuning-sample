set -e

sleep 3600

BASE_MODEL="answerdotai/ModernBERT-base"
for seed in {1,2,3,4,5}
do
    OUTPUT_DIR="ft/ModernBERT-base-seed-$seed"
    torchrun --nproc_per_node=8 --master_port 29501 train_chem.py \
        --base_model $BASE_MODEL \
        --output_dir $OUTPUT_DIR \
        --train_dataset "BASF-AI/dolma-chem-only-query-generated" \
        --seed $seed

    CUDA_VISIBLE_DEVICES=0 python eval_chem.py \
        --base_model $BASE_MODEL \
        --output_dir $OUTPUT_DIR \
        --max_active_dims 128 &
done

for vocab_size in {10000,20000,30000,40000,50000}
do
    for seed in {1,2,3,4,5}
    do
        BASE_MODEL="mlm/mdbert-corpus-sub_$vocab_size/checkpoint-3000"

        cd ..
        python probe_activation_percentiles.py \
            --model_id chem/$BASE_MODEL \
            --cut_percent 60 \
            --save_path chem/$BASE_MODEL \
            --include_zeros \
            --data_file chem/dolma-chem-only-query-generated.json \
            --num_docs 6000
        cd chem

        OUTPUT_DIR="ft/mdbert-corpus-sub_$vocab_size-seed-$seed"
        torchrun --nproc_per_node=8 --master_port 29501 train_chem.py \
            --base_model $BASE_MODEL \
            --output_dir $OUTPUT_DIR \
            --train_dataset "BASF-AI/dolma-chem-only-query-generated" \
            --seed $seed

        CUDA_VISIBLE_DEVICES=0 python eval_chem.py \
            --base_model $BASE_MODEL \
            --output_dir $OUTPUT_DIR \
            --max_active_dims 128 &
    done
done