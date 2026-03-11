# train_bge_m3_kldiv.py
# pip install -U sentence-transformers datasets accelerate torch

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    util,
)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

DATASET_TO_MTEB_TASK = {
    "arguana": "ArguAna",
    "dbpedia-entity": "DBPedia",
    "fiqa": "FiQA2018",
    "nfcorpus": "NFCorpus",
    "quora": "QuoraRetrieval",
    "scidocs": "SCIDOCS",
    "scifact": "SciFact",
    "trec-covid": "TRECCOVID",
}

def load_and_convert(jsonl_path: str, num_negatives: int, teacher_score_scale_factor: float, topN: int = None):
    queries, positives, labels = [], [], []
    negatives_cols = {f"negative{i+1}": [] for i in range(num_negatives)}

    n_skipped = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data_array = json.load(f)

    for ex in data_array:
        q = ex["query"]
        docs = ex["docs"]
        scores = ex["scores"]

        if not docs or not scores or len(docs) != len(scores):
            n_skipped += 1
            print(f"Skipping example because it has no docs or scores")
            assert False

        # 需要至少 1 个正例 + K 个负例
        if len(docs) < 1 + num_negatives:
            n_skipped += 1
            print(f"Skipping example because it has less than 1 + {num_negatives} docs")
            assert False

        # 选 teacher 分数最高的当 positive
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        pos_doc = docs[best_idx]
        pos_score = float(scores[best_idx])

        # 剩下的按 teacher 分数从高到低排序，作为 hardest negatives 列表
        rest = [(docs[i], float(scores[i])) for i in range(len(docs)) if i != best_idx]
        rest.sort(key=lambda x: x[1], reverse=True)

        # 如果设置了 topN，只保留前 topN 个 negatives
        if topN is not None:
            rest = rest[:topN]

        # 按 NUM_NEGATIVES 分块，每满一块就添加一条训练样本；不足一块时跳过，处理下一条
        offset = 0
        while offset + num_negatives <= len(rest):
            chunk = rest[offset : offset + num_negatives]
            offset += num_negatives

            queries.append(q)
            positives.append(pos_doc)
            # label 形状: [score_pos, score_neg1, ..., score_negK]
            labels.append(
                [pos_score * teacher_score_scale_factor]
                + [s * teacher_score_scale_factor for (_, s) in chunk]
            )
            for i, (neg_doc, _) in enumerate(chunk):
                negatives_cols[f"negative{i+1}"].append(neg_doc)

    data = {"query": queries, "positive": positives, **negatives_cols, "label": labels}
    ds = Dataset.from_dict(data)

    # Trainer/Loss 里希望 label 是 tensor
    ds = ds.map(lambda b: {"label": torch.tensor(b["label"], dtype=torch.float32)}, batched=False)
    print(f"Loaded {len(ds)} examples; skipped {n_skipped}")
    return ds

def parse_args():
    parser = argparse.ArgumentParser(description="Train a sentence transformer model using KL divergence distillation")

    parser.add_argument(
        "--dataset",
        type=str,
        default="nfcorpus",
        help="Dataset name, used to construct data_path and output_dir if not explicitly provided (default: nfcorpus)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the JSONL data file containing queries, docs, and scores (default: kd_samples/{dataset}.jsonl)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="BAAI/bge-m3",
        help="Name or path of the base model to finetune (default: BAAI/bge-m3)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for saving the trained model (default: bge-m3-kldiv-{dataset})"
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=2,
        help="Number of negative examples per query (default: 4)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for KL divergence loss (default: 1.0)"
    )
    parser.add_argument(
        "--use_dot",
        action="store_true",
        help="Use dot product instead of cosine similarity (default: use cosine)"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Per-device training batch size (default: 4)"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (default: 1)"
    )
    parser.add_argument(
        "--teacher_score_scale_factor",
        type=float,
        default=0.025,
        help="Scaling factor for teacher scores (default: 0.025)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate (default: 5e-6)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Maximum number of training steps. If set to positive value, overrides num_epochs (default: -1)"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio (default: 0.1)"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Logging steps (default: 50)"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep (default: 2)"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation after training"
    )
    parser.add_argument(
        "--topN",
        type=int,
        default=None,
        help="If set, only keep top N negatives (by teacher score) before generating training samples (default: None)"
    )
    parser.add_argument(
        "--freeze_first_half_layers",
        action="store_true",
        help="Freeze the first half of the model layers, only train the second half"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Construct data_path and output_dir from dataset if not explicitly provided
    if args.data_path is None:
        args.data_path = f"kd_samples/{args.dataset}.jsonl"
    if args.output_dir is None:
        args.output_dir = f"bge-m3-kldiv-{args.dataset}"

    train_dataset = load_and_convert(
        args.data_path,
        args.num_negatives,
        args.teacher_score_scale_factor,
        args.topN
    )

    model = SentenceTransformer(args.model_name, trust_remote_code=True)
    model.max_seq_length = args.max_seq_length

    # Freeze first half of layers if requested
    if args.freeze_first_half_layers:
        try:
            # Access the transformer model
            transformer = model[0].auto_model

            # Get encoder layers (works for BERT-like models)
            if hasattr(transformer, 'encoder') and hasattr(transformer.encoder, 'layer'):
                encoder_layers = transformer.encoder.layer
                num_layers = len(encoder_layers)
                half_layers = num_layers // 2

                print(f"Freezing first {half_layers} layers out of {num_layers} total layers")

                # Freeze embeddings
                if hasattr(transformer, 'embeddings'):
                    for param in transformer.embeddings.parameters():
                        param.requires_grad = False
                    print("Frozen embeddings")

                # Freeze first half of encoder layers
                for i in range(half_layers):
                    for param in encoder_layers[i].parameters():
                        param.requires_grad = False

                print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
                print(f"Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
            else:
                print("Warning: Could not find encoder.layer structure in model. No layers frozen.")
        except Exception as e:
            print(f"Warning: Failed to freeze layers: {e}")

    # DistillKLDivLoss：teacher labels softmax vs student log-softmax，然后 KL divergence
    similarity_fct = util.pairwise_cos_sim if not args.use_dot else util.pairwise_dot_score
    train_loss = losses.DistillKLDivLoss(
        model=model,
        similarity_fct=similarity_fct,
        temperature=args.temperature,  # 温度越大分布越"软"
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=torch.cuda.is_available(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    print(f"Saved to: {args.output_dir}")

    if not args.skip_eval:
        try:
            import mteb

            # Map dataset name to MTEB task name
            mteb_task = DATASET_TO_MTEB_TASK.get(args.dataset)
            if mteb_task is None:
                print(f"Warning: No MTEB task mapping found for dataset '{args.dataset}'. Skipping evaluation.")
            else:
                tasks = mteb.get_tasks(tasks=[mteb_task])
                evaluation = mteb.MTEB(tasks=tasks)

                # 1. 用变量 results 接收运行结果
                results = evaluation.run(
                    model,
                    eval_splits=["test"],
                    output_folder=None,
                    show_progress_bar=True,
                    encode_kwargs={
                        "batch_size": args.eval_batch_size,
                    },
                )

                # 2. 打印结果
                print("\n====== Evaluation Results ======")
                # results 是一个列表，包含每个任务的 MTEBResult 对象
                for result in results:
                    print(f"Task Name: {result.task_name}")

                    # 获取测试集的详细分数
                    test_scores = result.scores.get("test", {})

                    # 简单打印
                    print(test_scores)
        except ImportError:
            print("Warning: mteb not installed. Skipping evaluation.")
            print("Install with: pip install mteb")

if __name__ == "__main__":
    main()
