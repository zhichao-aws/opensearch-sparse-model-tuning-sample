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
from tokenizers.pre_tokenizers import Sequence, ByteLevel
from tokenizers.normalizers import BertNormalizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class PROCESSING(Enum):
    ALBERT = 1
    BERT = 2
    BERT_METASPACE = 3


use_data_file = True
processing = PROCESSING.BERT_METASPACE
os.environ["JSONL_LOCAL_FILES"] = (
    "/opt/dlami/nvme/dolma/wiki*,/opt/dlami/nvme/dolma/book*"
)
# os.environ["JSONL_LOCAL_FILES"] = "/opt/dlami/nvme/dolma/*"
# os.environ["JSONL_LOCAL_SUFFIX_MAX"] = "2"
output_dir = "modernbert-bpe-bert-wikibook"

data_file = "data/wikibook.ml128.jsonl"

if use_data_file:
    dataset = load_dataset(
        "json",
        data_files=[
            os.path.join("/home/ubuntu/dolma-jsonl/", f)
            for f in os.listdir("/home/ubuntu/dolma-jsonl/")
        ],
        split="train",
        num_proc=40,
    )
    texts = dataset["text"]

    def batch_iterator(batch_size=1000):
        for i in range(0, len(texts), batch_size):
            yield texts[i : i + batch_size]
else:
    dataset = load_dataset(
        "dataloader/jsonl_in_seq",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    def batch_iterator(
        batch_size=1000, num_workers=40, prefetch_factor=5, persistent_workers=True
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
        [bert_tokenizer.backend_tokenizer.pre_tokenizer, ByteLevel()]
    )

trainer = trainers.BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
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
