import torch
from deepfocus.focus import get_overlapping_tokens
from transformers import AutoModelForMaskedLM, AutoTokenizer

fuzzy = False
save = True
target_is_bpe = True
source_model_id = "answerdotai/ModernBERT-base"
target_model_id = "modernbert-bpe-bert-10k"
save_name = "modernbert-bpe-bert-10k-focus"

source_tokenizer = AutoTokenizer.from_pretrained(source_model_id)
target_tokenizer = AutoTokenizer.from_pretrained(target_model_id)
if target_is_bpe and target_tokenizer.backend_tokenizer.decoder is None:
    target_tokenizer.backend_tokenizer.decoder = (
        source_tokenizer.backend_tokenizer.decoder
    )
    print("reset target tokenizer. ", target_tokenizer.backend_tokenizer.decoder)

overlap, additional_tokens = get_overlapping_tokens(
    target_tokenizer,
    source_tokenizer,
    match_symbols=True,
    exact_match_all=True,
    fuzzy_match_all=fuzzy,
)

target_overlap_tokens = []
source_overlap_tokens = []
for key, value in overlap.items():
    target_overlap_tokens.append(key)
    source_overlap_tokens.append(value.source[0].native_form)

print(len(target_overlap_tokens), len(source_overlap_tokens))

# Load pretrained models
source_model = AutoModelForMaskedLM.from_pretrained(
    source_model_id, trust_remote_code=True
)

# Original embeddings
source_emb = source_model.get_input_embeddings()
orig_weight = source_emb.weight.data  # (V_roberta, dim)

# target vocab size and embedding dim (use source dim)
V_target = len(target_tokenizer)
dim = orig_weight.shape[1]

# Overlap tokens (strings) and their IDs in each vocab
overlap_ids_source = [
    source_tokenizer.convert_tokens_to_ids(tok) for tok in source_overlap_tokens
]
overlap_ids_target = [
    target_tokenizer.convert_tokens_to_ids(tok) for tok in target_overlap_tokens
]

# Create new embedding matrix of shape (V_target, dim)
new_weight = torch.zeros(
    (V_target, orig_weight.shape[1]), dtype=orig_weight.dtype, device=orig_weight.device
)

# 1. Copy original embeddings for overlap tokens
torch_tensor_overlap_source = orig_weight[overlap_ids_source]  # (|O|, dim)
new_weight[overlap_ids_target] = torch_tensor_overlap_source

# 2. Compute embeddings for non-overlap tokens
# Identify new token IDs in target vocab not in overlap
all_target_ids = list(range(V_target))
new_ids = [i for i in all_target_ids if i not in set(overlap_ids_target)]

# Prepare tensors
source_overlap = torch_tensor_overlap_source  # (|O|, dim) in source space

# Mean of overlap embeddings in source space, replicated for all new ids
mean_source_overlap = source_overlap.mean(dim=0)  # (dim,)
new_rows = mean_source_overlap.repeat(len(new_ids), 1)  # (N_new, dim)

# Assign to new_weight
new_weight[new_ids] = new_rows

# Replace source's embedding layer with the new matrix
new_emb_layer = torch.nn.Embedding.from_pretrained(new_weight, freeze=False)

source_model.resize_token_embeddings(len(target_tokenizer))
source_model.set_input_embeddings(new_emb_layer)
source_model.set_output_embeddings(new_emb_layer)

source_model.config.bos_token_id = target_tokenizer.vocab["[CLS]"]
source_model.config.eos_token_id = target_tokenizer.vocab["[SEP]"]
source_model.config.pad_token_id = target_tokenizer.vocab["[PAD]"]
source_model.config.sep_token_id = target_tokenizer.vocab["[SEP]"]
source_model.config.cls_token_id = target_tokenizer.vocab["[CLS]"]

source_model.save_pretrained(save_name)
if save_name != target_model_id:
    target_tokenizer.save_pretrained(save_name)

print("Saved updated source model.")
