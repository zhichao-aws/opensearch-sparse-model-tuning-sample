#!/usr/bin/env bash
set -euo pipefail

# Run EMLM training (Expanded Masked Language Model)
# This script mirrors the style of repo-level `run_mlm.sh`, but runs `vocab_trans/run_emlm.py`.

# Always run from this script's directory so relative paths work.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -------------------------
# Hardware / batch sizing
# -------------------------
DEVICE=8
TOTAL_BS=2048
DEVICE_BS=256
GRADIENT_ACCUMULATION_STEPS=$((TOTAL_BS / DEVICE_BS / DEVICE))

# -------------------------
# Model / tokenizer choices
# -------------------------
# Base MLM checkpoint (input tokenizer stays the same as this model)
BASE_MODEL="alignment-modernbert-base-semantic"

# Target tokenizer (U vocab). IDs are taken from this tokenizer's vocab.
# Choose one that exists under `vocab_trans/` (or set to an HF repo / absolute path).
TARGET_TOKENIZER_PATH="bert-base-uncased"

# Whether to initialize an expanded decoder from the base MLM head (mean-pooling wordpiece rows).
# - true: start from BASE_MODEL and build expanded head for TARGET_TOKENIZER_PATH
# - false: resume/finetune from an existing EMLM checkpoint in MODEL_NAME_OR_PATH
INIT_FROM_BASE=false

# Dataset (relative to vocab_trans/)
TRAIN_FILE="../data/wikibook.ml128.jsonl"

# -------------------------
# Training schedule
# -------------------------
MAX_SEQ_LENGTH=128
MLM_STEPS=(20000)
MASTER_PORT=29504

TERM_MASK_PROB=0.15
MAX_TERM_WORDPIECES=8
LABEL_ALL_SUBTOKENS=true
DROP_TERMS_WITH_UNK=true

for STEP in "${MLM_STEPS[@]}"; do
  OUT_DIR="emlm/${BASE_MODEL}-U$(basename "${TARGET_TOKENIZER_PATH}")-${STEP}"

  INIT_FLAGS=()
  if [ "${INIT_FROM_BASE}" = true ]; then
    INIT_FLAGS+=(--init_emlm_from_base_mlm)
  fi

  LABEL_FLAGS=()
  if [ "${LABEL_ALL_SUBTOKENS}" = true ]; then
    LABEL_FLAGS+=(--label_all_subtokens true)
  else
    LABEL_FLAGS+=(--label_all_subtokens false)
  fi

  DROP_UNK_FLAGS=()
  if [ "${DROP_TERMS_WITH_UNK}" = true ]; then
    DROP_UNK_FLAGS+=(--drop_terms_with_unk true)
  else
    DROP_UNK_FLAGS+=(--drop_terms_with_unk false)
  fi

  torchrun --nproc_per_node="${DEVICE}" --master_port "${MASTER_PORT}" run_emlm.py \
    --model_name_or_path "${BASE_MODEL}" \
    --tokenizer_name answerdotai/modernbert-base \
    --train_file "${TRAIN_FILE}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --target_tokenizer_path "${TARGET_TOKENIZER_PATH}" \
    --term_mask_prob "${TERM_MASK_PROB}" \
    --max_term_wordpieces "${MAX_TERM_WORDPIECES}" \
    "${LABEL_FLAGS[@]}" \
    "${DROP_UNK_FLAGS[@]}" \
    "${INIT_FLAGS[@]}" \
    --per_device_train_batch_size "${DEVICE_BS}" \
    --per_device_eval_batch_size "${DEVICE_BS}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --do_train \
    --output_dir "${OUT_DIR}" \
    --dataloader_drop_last \
    --dataloader_num_workers 8 \
    --logging_steps 50 \
    --max_steps "${STEP}" \
    --save_steps "${STEP}" \
    --warmup_steps 4000 \
    --optim adamw_torch \
    --report_to tensorboard \
    --lr_scheduler_type cosine \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --overwrite_output_dir \
    --fp16 \
    --log_level info \
    --preprocessing_num_workers 20
done

