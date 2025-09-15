import json
import os
from enum import Enum

from datasets import load_dataset
from tokenizers import (
    AddedToken,
    Tokenizer,
    models,
    trainers,
)
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import ByteLevel, Sequence
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class PROCESSING(Enum):
    ALBERT = 1
    BERT = 2
    BERT_METASPACE = 3


use_data_file = True
processing = PROCESSING.BERT_METASPACE
# os.environ["JSONL_LOCAL_FILES"] = "/opt/dlami/nvme/dolma/wiki*,/opt/dlami/nvme/dolma/book*"
# os.environ["JSONL_LOCAL_SUFFIX_MAX"] = "5"
output_dir = "modernbert-bpe-bert-wbm"
data_files = [
    os.path.join("/home/ubuntu/tokenizer_corpus/", f)
    for f in os.listdir("/home/ubuntu/tokenizer_corpus/")
]
print(data_files)

if use_data_file:
    dataset = load_dataset(
        "json",
        data_files=data_files,
        split="train",
        num_proc=40,
    )

    def batch_iterator(batch_size=2000):
        if len(dataset) < 1e8:
            texts = dataset["text"]
            for i in range(0, len(texts), batch_size):
                yield texts[i : i + batch_size]
        else:
            print("batching")
            batched_dataset = dataset.batch(batch_size)
            for batch in batched_dataset:
                yield batch["text"]

else:
    dataset = load_dataset(
        "dataloader/jsonl_in_seq",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    def batch_iterator(
        batch_size=1000, num_workers=10, prefetch_factor=5, persistent_workers=True
    ):
        # Only keep the text column to avoid decoding the rest of the columns unnecessarily
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            batch_size=batch_size,
            persistent_workers=persistent_workers,
        )

        for batch in dataloader:
            yield batch["text"]


albert_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
mdbert_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tokenizer = Tokenizer(models.BPE())

if processing == PROCESSING.ALBERT:
    tokenizer.normalizer = albert_tokenizer.backend_tokenizer.normalizer
    tokenizer.pre_tokenizer = mdbert_tokenizer.backend_tokenizer.pre_tokenizer
    tokenizer.pre_tokenizer.add_prefix_space = True
elif processing == PROCESSING.BERT_METASPACE:
    tokenizer.normalizer = BertNormalizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=True,
        lowercase=True,
    )
    tokenizer.pre_tokenizer = Sequence(
        # bert pre_tokenizer = Sequence([WhitespaceSplit(), Punctuation(behavior="isolated")])
        [bert_tokenizer.backend_tokenizer.pre_tokenizer, ByteLevel()]
    )
elif processing == PROCESSING.BERT:
    tokenizer.normalizer = BertNormalizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=True,
        lowercase=True,
    )
    tokenizer.pre_tokenizer = Sequence(
        [
            bert_tokenizer.backend_tokenizer.pre_tokenizer,
            ByteLevel(add_prefix_space=False),
        ]
    )

trainer = trainers.BpeTrainer(
    vocab_size=30000,
    min_frequency=10,
    initial_alphabet=ByteLevel.alphabet(),
    special_tokens=list(mdbert_tokenizer.special_tokens_map.values()),
)

tokenizer.train_from_iterator(
    batch_iterator(), trainer=trainer, length=len(dataset) if use_data_file else None
)

# Load ModernBERT added_tokens and add them with identical flags
mdbert_tokenizer_path = "modernbert-base"
mdbert_tokenizer.save_pretrained(mdbert_tokenizer_path)
mdbert_tokenizer_json_path = os.path.join(mdbert_tokenizer_path, "tokenizer.json")
with open(mdbert_tokenizer_json_path, "r", encoding="utf-8") as f:
    mdbert_tokenizer_json = json.load(f)

added_tokens_entries = mdbert_tokenizer_json.get("added_tokens", [])
existing_vocab_tokens = set(Tokenizer.get_vocab(tokenizer).keys())
builtin_specials = set(list(mdbert_tokenizer.special_tokens_map.values()))
special_added_tokens = []
regular_added_tokens = []
for entry in added_tokens_entries:
    content = entry.get("content", "")
    if not content:
        continue
    # Skip tokens already present or basic specials provided to trainer
    if content in existing_vocab_tokens or content in builtin_specials:
        continue
    token_obj = AddedToken(
        content,
        single_word=entry.get("single_word", False),
        lstrip=entry.get("lstrip", False),
        rstrip=entry.get("rstrip", False),
        normalized=entry.get("normalized", False),
        special=entry.get("special", False),
    )
    if entry.get("special", False):
        special_added_tokens.append(token_obj)
    else:
        regular_added_tokens.append(token_obj)

if special_added_tokens:
    tokenizer.add_special_tokens(special_added_tokens)
if regular_added_tokens:
    tokenizer.add_tokens(regular_added_tokens)

hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token=mdbert_tokenizer.unk_token,
    pad_token=mdbert_tokenizer.pad_token,
    cls_token=mdbert_tokenizer.cls_token,
    sep_token=mdbert_tokenizer.sep_token,
    mask_token=mdbert_tokenizer.mask_token,
)

hf_tokenizer.backend_tokenizer.pre_tokenizer.add_prefix_space = True
hf_tokenizer.model_max_length = mdbert_tokenizer.model_max_length

# Align post-processor with ModernBERT but bind to current special token ids
cls_tok = hf_tokenizer.cls_token
sep_tok = hf_tokenizer.sep_token
mask_tok = hf_tokenizer.mask_token
pad_tok = hf_tokenizer.pad_token
unk_tok = hf_tokenizer.unk_token

cls_id = hf_tokenizer.cls_token_id
sep_id = hf_tokenizer.sep_token_id
mask_id = hf_tokenizer.mask_token_id
pad_id = hf_tokenizer.pad_token_id
unk_id = hf_tokenizer.unk_token_id

special_token_pairs = []
for tok, tid in [
    (cls_tok, cls_id),
    (sep_tok, sep_id),
    (mask_tok, mask_id),
    (pad_tok, pad_id),
    (unk_tok, unk_id),
]:
    if tok is not None and tid is not None:
        special_token_pairs.append((tok, tid))

template = TemplateProcessing(
    single=f"{cls_tok}:0 $A:0 {sep_tok}:0",
    pair=f"{cls_tok}:0 $A:0 {sep_tok}:0 $B:0 {sep_tok}:0",
    special_tokens=special_token_pairs,
)

hf_tokenizer.backend_tokenizer.post_processor = template
hf_tokenizer.save_pretrained(output_dir)
tokenizer.save(os.path.join(output_dir, "original_config.json"))

with open(os.path.join(output_dir, "tokenizer_config.json")) as f:
    tokenizer_config = json.load(f)
tokenizer_config["add_prefix_space"] = True
tokenizer_config["model_input_names"] = ["input_ids", "attention_mask"]
tokenizer_config["clean_up_tokenization_spaces"] = True

with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
    json.dump(tokenizer_config, f, indent=4)

# fix special tokens
