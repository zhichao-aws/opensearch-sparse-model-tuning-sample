import json

from datasets import load_dataset
from tokenizers import (
    AddedToken,
    Tokenizer,
    models,
    trainers,
)
from tokenizers.pre_tokenizers import ByteLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import os


def batch_iterator(batch_size=1000):
    # Only keep the text column to avoid decoding the rest of the columns unnecessarily
    tok_dataset = dataset.select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        yield batch["text"]


use_data_file = True
data_file = "/home/zhichaog/neural-sparse/data/wikibook.ml128.jsonl"
data_name = "dataloader/jsonl_in_seq"
os.environ["JSONL_LOCAL_FILES"] = "/opt/dlami/nvme/dolma/*"

if use_data_file:
    dataset = load_dataset(
        "json",
        data_files=data_file,
        split="train",
    )
else:
    dataset = load_dataset(
        data_name,
        split="train",
    )

# dataset = dataset.select(range(3000))

albert_tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
mdbert_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = albert_tokenizer.backend_tokenizer.normalizer
tokenizer.pre_tokenizer = mdbert_tokenizer.backend_tokenizer.pre_tokenizer
tokenizer.pre_tokenizer.add_prefix_space = True
trainer = trainers.BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    initial_alphabet=ByteLevel.alphabet(),
    special_tokens=list(mdbert_tokenizer.special_tokens_map.values()),
)

tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(dataset))

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

tokenizer.save("modernbert-bpe.json")

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="modernbert-bpe.json",
    unk_token=mdbert_tokenizer.unk_token,
    pad_token=mdbert_tokenizer.pad_token,
    cls_token=mdbert_tokenizer.cls_token,
    sep_token=mdbert_tokenizer.sep_token,
    mask_token=mdbert_tokenizer.mask_token,
)

tokenizer.backend_tokenizer.pre_tokenizer.add_prefix_space = True
tokenizer.model_input_names = mdbert_tokenizer.model_input_names
tokenizer.model_max_length = mdbert_tokenizer.model_max_length
tokenizer.save_pretrained("modernbert-bpe-1")
