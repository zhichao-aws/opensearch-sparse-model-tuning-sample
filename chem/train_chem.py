import argparse
import os
import random
from dataclasses import dataclass

import torch
from datasets import load_dataset

# -------- Sentence-Transformers sparse training imports (v5) ----------
from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder import (
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
    losses,
)

# BatchSamplers imports may vary slightly between versions, making it compatible
try:
    from sentence_transformers.training_args import BatchSamplers
except Exception:
    try:
        from sentence_transformers.trainer import BatchSamplers
    except Exception:
        BatchSamplers = None


@dataclass
class Config:
    # Training data: recommended to use large-scale "synthetic query-paragraph pairs in the chemical domain" (1M+ level)
    # You can also replace it with other "query-doc pair" datasets
    train_dataset: str = "BASF-AI/dolma-chem-only-query-generated"
    train_split: str = "train"

    # Initialize a SPLADE (fill-mask model) from a standard BERT
    base_model: str = "mlm_model-1000-b3"
    # Hyperparameters (you can adjust according to computing power/data)
    output_dir: str = "./models/splade-mlm-1000-b3-chem"

    # Training data size (recommend smaller for testing; can be increased for formal experiments)
    max_train_samples: int = 200_000
    max_eval_samples: int = 5_000

    num_train_epochs: int = 1
    train_batch_size: int = 16
    eval_batch_size: int = 16
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    fp16: bool = True

    # SPLADE L1 regularization (common practice in papers/implementations: query reg small or 0, doc reg slightly larger)
    query_reg: float = 0.0
    doc_reg: float = 3e-3

    seed: int = 12

    # MTEB tasks
    mteb_tasks = ["ChemHotpotQARetrieval", "ChemNQRetrieval"]
    mteb_eval_splits = ["test"]
    mteb_out: str = "./results_mteb_chem"


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def guess_pair_columns(column_names):
    """
    Try to automatically guess the names of "query column" and "doc/passage column" in the training set.
    If it cannot be guessed, you will need to specify it manually (modify query_col / doc_col below).
    """
    query_cands = [
        "query",
        "question",
        "generated_query",
        "instruction",
        "title_query",
        "q",
    ]
    doc_cands = [
        "paragraph",
        "passage",
        "document",
        "context",
        "text",
        "answer",
        "abstract",
        "content",
        "d",
    ]

    q = next((c for c in query_cands if c in column_names), None)
    d = next((c for c in doc_cands if c in column_names), None)
    return q, d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model", type=str, default=None, help="Base model to initialize from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="BASF-AI/dolma-chem-only-query-generated",
        help="Train dataset",
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=200_000, help="Max train samples"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    cfg = Config()
    if args.base_model:
        cfg.base_model = args.base_model
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.train_dataset:
        cfg.train_dataset = args.train_dataset
    if args.max_train_samples:
        cfg.max_train_samples = args.max_train_samples
    if args.seed:
        cfg.seed = args.seed
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # 1) Load training dataset
    ds = load_dataset(cfg.train_dataset, split=cfg.train_split)
    print("Loaded:", ds)
    print("Columns:", ds.column_names)

    # 2) Pick (query, doc) columns
    query_col, doc_col = guess_pair_columns(ds.column_names)

    # If you can't guess at this step (query_col/doc_col is None), manually change to your dataset's actual column names
    if query_col is None or doc_col is None:
        raise ValueError(
            f"Cannot infer query/doc columns from {ds.column_names}. "
            f"Please set query_col/doc_col manually."
        )

    print(f"Using query_col={query_col}, doc_col={doc_col}")

    # 3) Subsample + split train/eval
    if cfg.max_train_samples is not None and cfg.max_train_samples < len(ds):
        ds = ds.shuffle(seed=cfg.seed).select(
            range(cfg.max_train_samples + cfg.max_eval_samples)
        )

    split = ds.train_test_split(
        test_size=min(cfg.max_eval_samples, len(ds) // 20), seed=cfg.seed
    )
    train_ds = split["train"]
    eval_ds = split["test"]

    # 4) Rename columns to what SparseMultipleNegativesRankingLoss expects: anchor / positive
    #    (anchor=queries, positive=docs)
    keep_cols = {query_col: "anchor", doc_col: "positive"}
    drop_cols = [c for c in train_ds.column_names if c not in keep_cols]

    train_ds = train_ds.rename_columns(keep_cols).remove_columns(drop_cols)
    eval_ds = eval_ds.rename_columns(keep_cols).remove_columns(drop_cols)

    print("Train example:", train_ds[0])

    # 5) Init SPLADE from a plain BERT MLM checkpoint
    model = SparseEncoder(cfg.base_model)
    # Retrieval usually uses dot; can run without setting, but explicit setting is recommended
    model.similarity_fn_name = "dot"
    model.max_seq_length = 512

    # 6) Loss: SPLADE regularizer + in-batch negatives
    loss = losses.SpladeLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=cfg.query_reg,
        document_regularizer_weight=cfg.doc_reg,
    )

    # 7) Training args
    args = SparseEncoderTrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16,
        bf16=False,
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=2,
        logging_steps=200,
        run_name=os.path.basename(cfg.output_dir.rstrip("/")),
        # MultipleNegativesRankingLoss series usually recommend no duplicates within a batch
        **(
            {"batch_sampler": BatchSamplers.NO_DUPLICATES}
            if BatchSamplers is not None
            else {}
        ),
    )

    # 8) Train
    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=loss,
    )
    trainer.train()

    # 9) Save
    final_dir = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(final_dir)
    print("Saved model to:", final_dir)


if __name__ == "__main__":
    main()
