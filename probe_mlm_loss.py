import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="只计算 MLM loss / perplexity（BeIR corpus）。")
    parser.add_argument("--model_id", type=str, required=True, help="模型路径或 HF ID")
    parser.add_argument(
        "--tokenizer_id",
        type=str,
        default="bert-base-uncased",
        help="tokenizer 路径或 HF ID",
    )
    parser.add_argument("--dataset", type=str, default="msmarco", help="BeIR 数据集名")
    parser.add_argument("--data_file", type=str, default=None, help="本地 jsonl 文件路径，若指定则优先使用此文件")
    parser.add_argument("--num_docs", type=int, default=2000, help="取前多少篇文档")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    if args.data_file:
        print(f"Loading data from file: {args.data_file}")
        # load_dataset("json", data_files=...) 默认生成 train split
        corpus = load_dataset("json", data_files=args.data_file, split="train")
    else:
        print(f"Loading BeIR/{args.dataset} corpus...")
        corpus = load_dataset(f"BeIR/{args.dataset}", "corpus", split="corpus")

    n = min(args.num_docs, len(corpus))
    print(f"Selecting first {n} documents...")
    dataset = corpus.select(range(n))

    print(f"Loading tokenizer: {args.tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    print(f"Loading model: {args.model_id}")
    model = AutoModelForMaskedLM.from_pretrained(args.model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def preprocess_function(examples):
        # 和你原脚本保持一致：只用 text
        texts = [t for t in examples["text"]]
        return tokenizer(texts, truncation=True, max_length=args.max_length)

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    dataloader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )

    total_loss = 0.0
    total_batches = 0

    print("Calculating MLM loss...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    print("\nResults:")
    print(f"Processed {n} documents.")
    print(f"Average MLM Loss: {avg_loss:.6f}")
    print(f"Perplexity: {ppl:.6f}")


if __name__ == "__main__":
    main()
