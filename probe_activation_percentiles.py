import argparse
import random
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sparse_activation(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """模仿 scripts/model/sparse_encoders.py 的 encode：maxpool + log1p(relu)."""
    # logits: (B, L, V), attention_mask: (B, L)
    masked_logits = logits.masked_fill((attention_mask == 0).unsqueeze(-1), -torch.inf)
    values, _ = torch.max(masked_logits, dim=1)  # (B, V)
    return values


def parse_percentiles(s: str) -> List[float]:
    out: List[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _maybe_sample_1d(values_1d: torch.Tensor, k: int) -> torch.Tensor:
    """从 1D tensor 里做带放回采样 k 个（GPU 友好）。"""
    n = values_1d.numel()
    if n == 0:
        return values_1d
    k = min(k, n)
    idx = torch.randint(0, n, (k,), device=values_1d.device)
    return values_1d[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "正常输入（不做 MLM mask）下，统计 activation 的 percentile。"
            "默认统计 sparse maxpool+log1p(relu) 后的非零激活分布，并额外统计输入 token 对应 logit 分布。"
        )
    )
    parser.add_argument("--model_id", type=str, required=True, help="模型路径或 HF ID")
    parser.add_argument(
        "--tokenizer_id",
        type=str,
        default="bert-base-uncased",
        help="tokenizer 路径或 HF ID",
    )
    parser.add_argument("--dataset", type=str, default="msmarco", help="BeIR 数据集名")
    parser.add_argument("--num_docs", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # percentile 相关
    parser.add_argument(
        "--percentiles",
        type=str,
        default="10,20,30,40,50,60,70,80,90,95,99,99.5,99.9",
        help="逗号分隔，例如 50,90,99,99.9",
    )
    parser.add_argument(
        "--sample_per_batch",
        type=int,
        default=20000,
        help="每个 batch 对每个统计项采样多少个值（带放回）。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1_000_000,
        help="每个统计项最多保留多少采样点（到上限后就不再追加）。",
    )
    parser.add_argument(
        "--include_zeros",
        action="store_true",
        help="是否把 0 激活也纳入 sparse activation 分布（默认只统计 >0）。",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    percentiles = parse_percentiles(args.percentiles)

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
        texts = [t for t in examples["text"]]
        return tokenizer(texts, truncation=True, max_length=args.max_length)

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    dataloader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )

    # 采样缓存（只存 CPU float32）
    samples: Dict[str, List[np.ndarray]] = {
        "sparse_activation": [],
        "input_token_logit": [],
    }

    total_seen = {
        "sparse_activation": 0,
        "input_token_logit": 0,
    }

    def _append_samples(key: str, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        if total_seen[key] >= args.max_samples:
            return
        x_cpu = x.detach().to("cpu", dtype=torch.float32).numpy()
        remaining = args.max_samples - total_seen[key]
        if x_cpu.size > remaining:
            x_cpu = x_cpu[:remaining]
        samples[key].append(x_cpu)
        total_seen[key] += int(x_cpu.size)

    print("Running forward (no masking) and sampling activations...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)  # no labels -> no loss
            logits = outputs.logits  # (B, L, V)
            attention_mask = batch["attention_mask"]

            # 1) sparse activation (doc-level pooled)
            d_rep = sparse_activation(logits, attention_mask)  # (B, V)
            if args.include_zeros:
                rep_vals = d_rep.reshape(-1)
            else:
                rep_vals = d_rep[d_rep > 0]

            rep_sample = _maybe_sample_1d(rep_vals, args.sample_per_batch)
            _append_samples("sparse_activation", rep_sample)

            # 2) input token 对应 logit（逐位置，排除 padding）
            # logits.gather(2, input_ids.unsqueeze(-1)) -> (B, L, 1)
            token_logits = logits.gather(2, batch["input_ids"].unsqueeze(-1)).squeeze(-1)
            token_logits = token_logits[attention_mask == 1]
            tok_sample = _maybe_sample_1d(token_logits.reshape(-1), args.sample_per_batch)
            _append_samples("input_token_logit", tok_sample)

    print("\nPercentiles:")
    for key, chunks in samples.items():
        if not chunks:
            print(f"- {key}: (no samples)")
            continue
        arr = np.concatenate(chunks, axis=0)
        stats = {
            "count": int(arr.size),
            "mean": float(arr.mean()) if arr.size else 0.0,
            "std": float(arr.std()) if arr.size else 0.0,
        }
        pct = {p: float(np.percentile(arr, p)) for p in percentiles}

        print(f"\n[{key}]")
        print(f"count={stats['count']} mean={stats['mean']:.6f} std={stats['std']:.6f}")
        for p in percentiles:
            print(f"P{p}: {pct[p]:.6f}")


if __name__ == "__main__":
    main()
