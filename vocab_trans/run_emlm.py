#!/usr/bin/env python
# coding=utf-8
"""
Train EMLM (Expanded Masked Language Model) with:
- base tokenizer: RoBERTa byte-level BPE (input_ids space)
- target tokenizer: bert-base-uncased WordPiece (output vocab U)
- masking: find spans in base-tokenized input_ids that match sequences derived from U tokens,
          then mask those spans and predict the U token id.

Model requirement:
- model forward supports labels shaped [bs, seq_len] with ignore_index=-100
- logits shaped [bs, seq_len, target_vocab_size]
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Optional, Tuple

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_xla_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from scripts.model.models import AlignmentMDBertForMaskedLM, AlignmentMDBertConfig

AutoConfig.register("alignment-modernbert", AlignmentMDBertConfig)
AutoModelForMaskedLM.register(AlignmentMDBertConfig, AlignmentMDBertForMaskedLM)

check_min_version("4.50.0")
require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

# -----------------------
# Args
# -----------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Checkpoint for initialization (AlignmentMDBertForMaskedLM)."}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Base tokenizer name/path. If None, defaults to model_name_or_path."},
    )
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache dir"})
    token: str = field(default=None)
    trust_remote_code: bool = field(default=False)


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default=None)
    validation_file: Optional[str] = field(default=None)
    overwrite_cache: bool = field(default=False)
    validation_split_percentage: Optional[int] = field(default=5)
    max_seq_length: Optional[int] = field(default=None)
    preprocessing_num_workers: Optional[int] = field(default=None)
    line_by_line: bool = field(default=False)
    pad_to_max_length: bool = field(default=False)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    streaming: bool = field(default=False)

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "streaming requires datasets>=2.0.0")
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")


@dataclass
class EMLMArguments:
    target_tokenizer_path: str = field(
        metadata={
            "help": (
                "Name/path of target tokenizer (U). Here you use bert-base-uncased. "
                "IDs follow that tokenizer's token IDs."
            )
        },
    )
    term_mask_prob: float = field(
        default=0.15,
        metadata={"help": "Mask ratio over number of matched terms (not over tokens)."},
    )
    ensure_at_least_one_term_mask: bool = field(
        default=True,
        metadata={"help": "If True and there are any matched terms, mask at least one term per sequence."},
    )
    label_all_subtokens: bool = field(
        default=True,
        metadata={
            "help": (
                "If True, write the same term_id label to all subtoken positions in the span. "
                "If False, only label the first subtoken and set others to -100."
            )
        },
    )
    max_term_wordpieces: int = field(
        default=8,
        metadata={"help": "Drop target terms whose base-tokenizer (RoBERTa BPE) length exceeds this value."},
    )
    drop_terms_with_unk: bool = field(
        default=True,
        metadata={"help": "Drop target terms whose base-tokenizer decomposition contains UNK."},
    )
    replace_prob_mask: float = field(default=0.8, metadata={"help": "Within masked span: replace with [MASK] prob."})
    replace_prob_random: float = field(default=0.1, metadata={"help": "Within masked span: replace with random token prob."})


# -----------------------
# Trie for span matching
# -----------------------
class _TrieNode:
    __slots__ = ("children", "term_id")

    def __init__(self):
        self.children: Dict[int, "_TrieNode"] = {}
        self.term_id: Optional[int] = None


class WordpieceTrie:
    def __init__(self):
        self.root = _TrieNode()

    def add(self, wp_ids: List[int], term_id: int):
        node = self.root
        for wid in wp_ids:
            node = node.children.setdefault(wid, _TrieNode())
        node.term_id = term_id

    def find_longest_at(
        self,
        ids: List[int],
        start: int,
        special_mask: Optional[List[int]] = None,
        pad_id: Optional[int] = None,
    ) -> Optional[Tuple[int, int, int]]:
        node = self.root
        best = None
        j = start
        n = len(ids)
        while j < n:
            if pad_id is not None and ids[j] == pad_id:
                break
            if special_mask is not None and special_mask[j] == 1:
                break
            nxt = node.children.get(ids[j])
            if nxt is None:
                break
            node = nxt
            j += 1
            if node.term_id is not None:
                best = (start, j, node.term_id)
        return best

    def greedy_non_overlapping_matches(
        self,
        ids: List[int],
        special_mask: Optional[List[int]] = None,
        pad_id: Optional[int] = None,
    ) -> List[Tuple[int, int, int]]:
        matches = []
        i = 0
        n = len(ids)
        while i < n:
            if pad_id is not None and ids[i] == pad_id:
                break
            if special_mask is not None and special_mask[i] == 1:
                i += 1
                continue
            m = self.find_longest_at(ids, i, special_mask=special_mask, pad_id=pad_id)
            if m is None:
                i += 1
                continue
            s, e, tid = m
            matches.append((s, e, tid))
            i = e
        return matches


# -----------------------
# Target tokenizer loader + mapping to base (RoBERTa BPE) ids
# -----------------------
def load_target_vocab_from_tokenizer(
    target_tokenizer,
    base_tokenizer,
    max_term_wordpieces: int,
    drop_terms_with_unk: bool,
) -> Tuple[List[str], List[List[int]]]:
    """
    Build target vocab (U) from a tokenizer:
      - terms[term_id] = token string from target_tokenizer vocab
      - term_wp_ids[term_id] = list of base_tokenizer token IDs (RoBERTa BPE) for a best-effort rendering of that token

    Specifically tuned for:
      target_tokenizer = bert-base-uncased (WordPiece, includes '##' continuation pieces)
      base_tokenizer   = roberta byte-level BPE (expects surface strings, sensitive to leading space)

    Heuristic mapping:
      - if tok startswith '##': core = tok[2:], try encode(core) then encode(" "+core)
      - else: try encode(" "+tok) then encode(tok)
    """
    vocab = target_tokenizer.get_vocab()
    if not vocab:
        raise ValueError("target_tokenizer.get_vocab() returned empty vocab.")

    max_id = max(vocab.values())
    target_vocab_size = max_id + 1

    terms: List[str] = [""] * target_vocab_size
    for tok, tid in vocab.items():
        if 0 <= tid < target_vocab_size:
            terms[tid] = tok

    base_unk = getattr(base_tokenizer, "unk_token_id", None)
    special_tokens = set(getattr(target_tokenizer, "all_special_tokens", []) or [])

    def _is_ok(ids: List[int]) -> bool:
        if not ids:
            return False
        if drop_terms_with_unk and base_unk is not None and base_unk in ids:
            return False
        if max_term_wordpieces is not None and len(ids) > max_term_wordpieces:
            return False
        return True

    def _encode_best_effort(tok: str) -> List[int]:
        if tok.startswith("[") and tok.endswith("]"):
            candidates = [tok]
        elif tok.startswith("##") and len(tok) > 2:
            core = tok[2:]
            candidates = [core, " " + core]
        else:
            candidates = [" " + tok, tok]

        for text in candidates:
            ids = base_tokenizer.encode(text, add_special_tokens=False)
            if _is_ok(ids):
                return [int(x) for x in ids]
        return []

    term_wp_ids: List[List[int]] = [[] for _ in range(target_vocab_size)]
    for tid, tok in enumerate(terms):
        if not tok:
            continue
        if tok in special_tokens:
            continue

        wp = _encode_best_effort(tok)
        if not wp:
            continue

        term_wp_ids[tid] = wp

    return terms, term_wp_ids


# -----------------------
# Data collator for EMLM
# -----------------------
class DataCollatorForExpandedMLM:
    def __init__(
        self,
        tokenizer,
        trie: WordpieceTrie,
        term_mask_prob: float,
        ensure_at_least_one_term_mask: bool,
        label_all_subtokens: bool,
        target_vocab_size: int,
        pad_to_multiple_of: Optional[int] = None,
        replace_prob_mask: float = 0.8,
        replace_prob_random: float = 0.1,
        ignore_index: int = -100,
    ):
        self.tokenizer = tokenizer
        self.trie = trie
        self.term_mask_prob = term_mask_prob
        self.ensure_at_least_one_term_mask = ensure_at_least_one_term_mask
        self.label_all_subtokens = label_all_subtokens
        self.target_vocab_size = target_vocab_size
        self.pad_to_multiple_of = pad_to_multiple_of
        self.replace_prob_mask = replace_prob_mask
        self.replace_prob_random = replace_prob_random
        self.ignore_index = ignore_index

        if self.tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer must have a [MASK] token for EMLM.")
        if not (0.0 <= replace_prob_mask <= 1.0 and 0.0 <= replace_prob_random <= 1.0):
            raise ValueError("replace_prob_* must be within [0,1].")
        if replace_prob_mask + replace_prob_random > 1.0 + 1e-6:
            raise ValueError("replace_prob_mask + replace_prob_random must be <= 1.0.")

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            examples,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        input_ids: torch.Tensor = batch["input_ids"]
        bs, seqlen = input_ids.shape

        special_tokens_mask = batch.get("special_tokens_mask", None)
        if special_tokens_mask is not None:
            special_tokens_mask = special_tokens_mask.tolist()

        pad_id = self.tokenizer.pad_token_id
        mask_id = self.tokenizer.mask_token_id

        labels = torch.full((bs, seqlen), fill_value=self.ignore_index, dtype=torch.long)

        for b in range(bs):
            ids_list = input_ids[b].tolist()
            stm = special_tokens_mask[b] if special_tokens_mask is not None else None

            matches = self.trie.greedy_non_overlapping_matches(ids_list, special_mask=stm, pad_id=pad_id)
            if len(matches) == 0:
                continue

            k = int(round(self.term_mask_prob * len(matches)))
            if self.ensure_at_least_one_term_mask:
                k = max(1, k)
            k = min(k, len(matches))
            if k <= 0:
                continue

            perm = torch.randperm(len(matches))
            chosen = [matches[i] for i in perm[:k].tolist()]

            for (s, e, term_id) in chosen:
                if term_id < 0 or term_id >= self.target_vocab_size:
                    continue

                # labels
                if self.label_all_subtokens:
                    for pos in range(s, e):
                        labels[b, pos] = term_id
                else:
                    labels[b, s] = term_id
                    for pos in range(s + 1, e):
                        labels[b, pos] = self.ignore_index

                # replacements
                for pos in range(s, e):
                    r = torch.rand(1).item()
                    if r < self.replace_prob_mask:
                        input_ids[b, pos] = mask_id
                    elif r < self.replace_prob_mask + self.replace_prob_random:
                        input_ids[b, pos] = int(torch.randint(low=0, high=len(self.tokenizer), size=(1,)).item())
                    else:
                        pass

        batch["labels"] = labels
        return batch


# -----------------------
# Main
# -----------------------
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, EMLMArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, emlm_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, emlm_args, training_args = parser.parse_args_into_dataclasses()

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training args: {training_args}")

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output dir ({training_args.output_dir}) exists and is not empty. Use --overwrite_output_dir."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming at {last_checkpoint}")

    set_seed(training_args.seed)

    # Load dataset
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file

        # infer builder
        ext = None
        for _, v in data_files.items():
            if isinstance(v, str):
                ext = v.split(".")[-1]
                break
        if ext == "txt":
            builder = "text"
        elif ext == "jsonl":
            builder = "json"
        else:
            builder = ext

        raw_datasets = load_dataset(
            builder,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                builder,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
            raw_datasets["train"] = load_dataset(
                builder,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )

    # Load base tokenizer (RoBERTa BPE)
    base_tok_name = model_args.tokenizer_name or model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        base_tok_name,
    )

    # Load target tokenizer (bert-base-uncased)
    target_tokenizer = AutoTokenizer.from_pretrained(
        emlm_args.target_tokenizer_path,
    )

    # Load target tokenizer vocab + build trie (U-vocab)
    terms, term_wp_ids = load_target_vocab_from_tokenizer(
        target_tokenizer=target_tokenizer,
        base_tokenizer=tokenizer,
        max_term_wordpieces=emlm_args.max_term_wordpieces,
        drop_terms_with_unk=emlm_args.drop_terms_with_unk,
    )
    target_vocab_size = len(terms)

    trie = WordpieceTrie()
    kept = 0
    for tid, wp in enumerate(term_wp_ids):
        if not wp:
            continue
        trie.add(wp, tid)
        kept += 1
    logger.info(f"Loaded target vocab size={target_vocab_size}, usable_terms={kept} (after filtering)")

    # Load model (allow mismatched decoder size if you changed target_vocab_size)
    cfg = AlignmentMDBertConfig.from_pretrained(model_args.model_name_or_path)
    cfg.target_vocab_size = target_vocab_size
    cfg.tie_word_embeddings = False

    model = AlignmentMDBertForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        config=cfg,
        ignore_mismatched_sizes=True,
    )

    # Tokenize datasets
    column_names = list(raw_datasets["train"].features) if training_args.do_train else list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            max_seq_length = 1024
    else:
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if data_args.line_by_line:
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            examples[text_column_name] = [l for l in examples[text_column_name] if l and not l.isspace()]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )

        with training_args.main_process_first(desc="tokenize"):
            map_kwargs = dict(
                batched=True,
                remove_columns=[text_column_name],
                desc="Tokenizing (line_by_line)",
            )
            if not data_args.streaming:
                map_kwargs.update(
                    load_from_cache_file=not data_args.overwrite_cache,
                    num_proc=data_args.preprocessing_num_workers,
                )
            tokenized = raw_datasets.map(tokenize_function, **map_kwargs)
    else:
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with training_args.main_process_first(desc="tokenize"):
            map_kwargs = dict(
                batched=True,
                remove_columns=column_names,
                desc="Tokenizing",
            )
            if not data_args.streaming:
                map_kwargs.update(
                    load_from_cache_file=not data_args.overwrite_cache,
                    num_proc=data_args.preprocessing_num_workers,
                )
            tokenized = raw_datasets.map(tokenize_function, **map_kwargs)

        def group_texts(examples):
            concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated[list(examples.keys())[0]])
            total_length = (total_length // max_seq_length) * max_seq_length
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated.items()
            }
            return result

        with training_args.main_process_first(desc="group"):
            map_kwargs = dict(
                batched=True,
                desc=f"Grouping into chunks of {max_seq_length}",
            )
            if not data_args.streaming:
                map_kwargs.update(
                    load_from_cache_file=not data_args.overwrite_cache,
                    num_proc=data_args.preprocessing_num_workers,
                )
            tokenized = tokenized.map(group_texts, **map_kwargs)

    train_dataset = None
    eval_dataset = None
    if training_args.do_train:
        train_dataset = tokenized["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

    if training_args.do_eval:
        eval_dataset = tokenized["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    # Metrics
    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        if labels.numel() == 0:
            return {"accuracy": 0.0}
        return metric.compute(predictions=preds, references=labels)

    # Data collator (EMLM)
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForExpandedMLM(
        tokenizer=tokenizer,
        trie=trie,
        term_mask_prob=emlm_args.term_mask_prob,
        ensure_at_least_one_term_mask=emlm_args.ensure_at_least_one_term_mask,
        label_all_subtokens=emlm_args.label_all_subtokens,
        target_vocab_size=target_vocab_size,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
        replace_prob_mask=emlm_args.replace_prob_mask,
        replace_prob_random=emlm_args.replace_prob_random,
        ignore_index=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval and not is_torch_xla_available() else None,
    )

    # Train
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset) if train_dataset is not None else 0
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Eval
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset) if eval_dataset is not None else 0
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except Exception:
            metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
