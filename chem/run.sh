BASE_MODEL="bert-base-uncased"

OUTPUT_DIR="ft/bert-base-uncased-chem1"

torchrun --nproc_per_node=8 --master_port 29501 train_chem.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --train_dataset "BASF-AI/dolma-chem-only-query-generated"

CUDA_VISIBLE_DEVICES=0 python eval_chem.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --max_active_dims 128 &

OUTPUT_DIR="ft/bert-base-uncased-chem2"
# torchrun --nproc_per_node=8 --master_port 29501 train_chem.py \
#     --base_model $BASE_MODEL \
#     --output_dir $OUTPUT_DIR \
#     --train_dataset "BASF-AI/ChemRxiv-Train-CC-BY"
CUDA_VISIBLE_DEVICES=1 python eval_chem.py \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --max_active_dims 128 &