import argparse
import random
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding

from scripts.model.models import AlignmentMDBertForMaskedLM, AlignmentMDBertConfig

AutoConfig.register("alignment-modernbert", AlignmentMDBertConfig)
AutoModelForMaskedLM.register(AlignmentMDBertConfig, AlignmentMDBertForMaskedLM)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sparse_activation(
    logits: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """For logits (B, L, V), perform padding mask and then max-pool along the token dimension to get (B, V).

    Note: This returns **original max logits** (can be negative), excluding relu / log1p.
    Subsequent statistics will default to filtering out values <= 0 (unless --include_zeros is passed).
    """
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
    """Sample k elements from a 1D tensor with replacement (GPU-friendly)."""
    n = values_1d.numel()
    if n == 0:
        return values_1d
    k = min(k, n)
    idx = torch.randint(0, n, (k,), device=values_1d.device)
    return values_1d[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate activation percentiles under normal input (no MLM mask). "
            "Defaults to statistics of non-zero activations after sparse maxpool+log1p(relu), with additional statistics for logit distribution corresponding to input tokens."
        )
    )
    parser.add_argument("--model_id", type=str, required=True, help="Model path or HF ID")
    parser.add_argument(
        "--tokenizer_id",
        type=str,
        default="bert-base-uncased",
        help="tokenizer path or HF ID",
    )
    parser.add_argument("--dataset", type=str, default="msmarco", help="BeIR dataset name")
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Local jsonl file path; if specified, this file is used with priority",
    )
    parser.add_argument("--num_docs", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    # percentile related
    parser.add_argument(
        "--percentiles",
        type=str,
        default="10,20,30,40,50,60,70,80,90,95,99,99.5,99.9",
        help="Comma-separated, e.g., 50,90,99,99.9",
    )
    parser.add_argument(
        "--sample_per_batch",
        type=int,
        default=20000,
        help="How many values to sample for each statistical item per batch (with replacement).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1_000_000,
        help="Maximum number of sampling points to keep for each statistical item (stop appending after reaching the limit).",
    )
    parser.add_argument(
        "--include_zeros",
        action="store_true",
        help=(
            "Whether to include sparse max-logits <= 0 (including 0 and negative values) in the sparse_activation distribution. "
            "Defaults to only counting values > 0 (equivalent to calculating percentiles on the 'conditional distribution: activation > 0')."
        ),
    )
    parser.add_argument(
        "--cut_percent",
        type=float,
        default=None,
        help=(
            "If set (e.g., 60), after printing percentiles, the P(cut_percent) value of the current sampling distribution "
            "will be subtracted from the bias of the output embedding, and the model can optionally be saved."
        ),
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save the modified model."
    )

    args = parser.parse_args()

    set_seed(args.seed)

    percentiles = parse_percentiles(args.percentiles)

    if args.data_file:
        print(f"Loading data from file: {args.data_file}")
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

    # Sample cache (CPU float32 only)
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

            # 2) input token logit (position-wise, excluding padding)
            # # logits.gather(2, input_ids.unsqueeze(-1)) -> (B, L, 1)
            # token_logits = logits.gather(2, batch["input_ids"].unsqueeze(-1)).squeeze(
            #     -1
            # )
            # token_logits = token_logits[attention_mask == 1]
            # tok_sample = _maybe_sample_1d(
            #     token_logits.reshape(-1), args.sample_per_batch
            # )
            # _append_samples("input_token_logit", tok_sample)

    print(
        "\nPercentiles: (Note: Stats printed here are before applying cut; sparse_activation defaults to only counting values > 0)"
    )
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

    if args.cut_percent is not None:
        key = "sparse_activation"
        if key in samples and samples[key]:
            arr = np.concatenate(samples[key], axis=0)
            cutoff_value = float(np.percentile(arr, args.cut_percent))
            print(
                f"\nApplying cut: subtracting P{args.cut_percent} ({cutoff_value:.6f}) from output embeddings bias..."
            )

            output_embeddings = model.get_output_embeddings()
            if (
                hasattr(output_embeddings, "bias")
                and output_embeddings.bias is not None
            ):
                # Subtract cutoff
                with torch.no_grad():
                    output_embeddings.bias.data -= cutoff_value
                print(f"Done. Subtracted {cutoff_value:.6f} from bias.")

                if args.save_path:
                    print(f"Saving modified model to {args.save_path}")
                    model.save_pretrained(args.save_path)
                    tokenizer.save_pretrained(args.save_path)
            else:
                print(
                    "Warning: Could not find bias in output embeddings (model.get_output_embeddings().bias)."
                )
        else:
            print(f"\nWarning: Cannot apply cut because no samples found for {key}.")


if __name__ == "__main__":
    main()
