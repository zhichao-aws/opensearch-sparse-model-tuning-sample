set -e

N_DEVICES=8
OUTPUT_DIR="output/test"

torchrun --nproc_per_node=${N_DEVICES} demo_train_data.py \
  --model_name_or_path opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill \
  --inf_free true \
  --idf_path idf.json \
  --beir_dir data/beir \
  --beir_datasets scifact

torchrun --nproc_per_node=${N_DEVICES} train_ir.py configs/config_infonce.yaml
torchrun --nproc_per_node=${N_DEVICES} evaluate_beir.py configs/config_infonce.yaml