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

# BatchSamplers 的 import 在不同版本位置可能略有差异，做个兼容
try:
    from sentence_transformers.training_args import BatchSamplers
except Exception:
    try:
        from sentence_transformers.trainer import BatchSamplers
    except Exception:
        BatchSamplers = None


@dataclass
class Config:
    # 训练数据：推荐用“化学领域合成 query–paragraph 对”的大数据（1M+ 级别）
    # 你也可以换成别的“query-doc pair”数据集
    train_dataset: str = "BASF-AI/dolma-chem-only-query-generated"
    train_split: str = "train"

    # 从普通 BERT 初始化一个 SPLADE（fill-mask 模型）
    base_model: str = "mlm_model-1000-b3"
    # 超参（你可以按算力/数据再调）
    output_dir: str = "./models/splade-mlm-1000-b3-chem"

    # 训练数据量（先跑通建议用小一点；正式实验可以拉大）
    max_train_samples: int = 200_000
    max_eval_samples: int = 5_000

    num_train_epochs: int = 1
    train_batch_size: int = 16
    eval_batch_size: int = 16
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    fp16: bool = True

    # SPLADE L1 正则（论文/实现里常见做法：query 正则小或 0，doc 正则略大）
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
    尽量自动猜训练集里“query 列”和“doc/passage 列”叫什么。
    猜不到就让你手动指定（改下面 query_col / doc_col）。
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

    # 如果你这一步猜不到（query_col/doc_col 是 None），就手动改成你数据集真实列名
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
    #    （anchor=queries, positive=docs）
    keep_cols = {query_col: "anchor", doc_col: "positive"}
    drop_cols = [c for c in train_ds.column_names if c not in keep_cols]

    train_ds = train_ds.rename_columns(keep_cols).remove_columns(drop_cols)
    eval_ds = eval_ds.rename_columns(keep_cols).remove_columns(drop_cols)

    print("Train example:", train_ds[0])

    # 5) Init SPLADE from a plain BERT MLM checkpoint
    model = SparseEncoder(cfg.base_model)
    # Retrieval 常用 dot；不设也能跑，但建议显式设一下
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
        # MultipleNegativesRankingLoss 系列通常建议 batch 内不重复
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
