N_DEVICES=8
OUTPUT_DIR="output/test"

torchrun --nproc_per_node=${N_DEVICES} demo_train_data.py \
  --model_name_or_path opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill \
  --inf_free true \
  --idf_path idf.json \
  --beir_dir data/beir \
  --beir_datasets scifact

torchrun --nproc_per_node=${N_DEVICES} train_ir.py configs/config_infonce.yaml

for step in {500,1000,1500,2000}
do
torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py \
  --model_name_or_path ${OUTPUT_DIR}/checkpoint-${step} \
  --inf_free true \
  --idf_path idf.json \
  --output_dir ${OUTPUT_DIR} \
  --log_level info \
  --beir_datasets scifact \
  --per_device_eval_batch_size 50
done