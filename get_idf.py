import argparse
import json
import math
import os
from collections import Counter

import datasets
from transformers import AutoTokenizer


def transform_str(s):
    try:
        s = s.encode("latin1").decode("utf-8")
        return s
    except Exception:
        return s


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute IDF from BeIR/msmarco corpus using a Hugging Face tokenizer"
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        type=str,
        required=True,
        help="Hugging Face tokenizer id or local path (e.g., 'bert-base-uncased')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output JSON path for IDF mapping (token_id -> idf)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=30,
        help="Number of processes for datasets.map",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for batched tokenization",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    special_ids = set(tokenizer.all_special_ids or [])

    # 1) load dataset
    msmarco_corpus = datasets.load_dataset("BeIR/msmarco", "corpus")["corpus"]

    # 2) fix occasional text encoding issues
    msmarco_corpus = msmarco_corpus.map(
        lambda x: {"text": transform_str(x["text"])},
        num_proc=30,
        desc="Normalizing text",
    )

    # 3) tokenize in parallel and aggregate DF per batch to reduce later accumulation cost
    def _tokenize_batch(batch):
        encodings = tokenizer(
            batch["text"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        batch_df = Counter()
        valid_docs = 0
        for ids in encodings["input_ids"]:
            unique_ids = {tid for tid in ids if tid not in special_ids}
            if unique_ids:
                batch_df.update(unique_ids)
                valid_docs += 1

        batch_len = len(batch["text"])
        if len(batch_df) == 0:
            # no valid tokens in this batch
            df_keys = [[] for _ in range(batch_len)]
            df_vals = [[] for _ in range(batch_len)]
            valid_docs_list = [0 for _ in range(batch_len)]
        else:
            keys_list = list(batch_df.keys())
            vals_list = [batch_df[k] for k in keys_list]
            # put aggregated stats only on the first example to avoid duplication
            df_keys = [keys_list] + [[] for _ in range(batch_len - 1)]
            df_vals = [vals_list] + [[] for _ in range(batch_len - 1)]
            valid_docs_list = [valid_docs] + [0 for _ in range(batch_len - 1)]

        return {"df_keys": df_keys, "df_vals": df_vals, "valid_docs": valid_docs_list}

    tokenized = msmarco_corpus.map(
        _tokenize_batch,
        batched=True,
        batch_size=args.batch_size,
        remove_columns=[col for col in msmarco_corpus.column_names if col != "text"],
        num_proc=args.num_proc,
        desc="Tokenizing corpus",
    )

    # 4) accumulate DF from batch-level aggregates
    df_counter: Counter[int] = Counter()
    df_keys_col = tokenized["df_keys"]
    df_vals_col = tokenized["df_vals"]
    valid_docs_col = tokenized["valid_docs"]
    for keys, vals in zip(df_keys_col, df_vals_col):
        if keys:
            df_counter.update(dict(zip(keys, vals)))

    # use number of valid docs (documents that have at least one non-special token)
    num_docs = sum(valid_docs_col)
    print(num_docs)

    # 5) compute IDF and write JSON (only tokens that appeared)
    idf_by_token_id = {
        tokenizer._convert_id_to_token(tid): math.log(
            1 + (num_docs - df + 0.5) / (df + 0.5)
        )
        for tid, df in df_counter.items()
        if df > 0
    }

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(idf_by_token_id, f, ensure_ascii=False)

    print(
        f"Wrote IDF for {len(idf_by_token_id)} tokens from {num_docs} documents -> {args.output}"
    )


if __name__ == "__main__":
    main()
