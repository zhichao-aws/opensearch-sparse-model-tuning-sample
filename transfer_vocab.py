from tokenizers.pre_tokenizers import Split, Sequence, WhitespaceSplit
from tokenizers.normalizers import BertNormalizer
from tokenizers import Regex
from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str, default="output/paper/bi/bert-vocab-sb-tn-pmi-VT-b5-20000/final/checkpoint-150000", help="Base model ID or path")
    parser.add_argument("--dataset_name", type=str, default="trec-covid", help="Dataset name (e.g. trec-covid)")
    parser.add_argument("--save_path", type=str, default="test", help="Path to save the model and tokenizer")
    args = parser.parse_args()

    base = args.base_model_id
    model = AutoModelForMaskedLM.from_pretrained(base)
    tokenizer = AutoTokenizer.from_pretrained(base)
    base_vocab = tokenizer.vocab

    dataset = load_dataset(
        f"BEIR/{args.dataset_name}", "corpus", split="corpus", trust_remote_code=True
    )

    normalizer = BertNormalizer(lowercase=True, strip_accents=True)
    custom_split_pattern = Regex(r"[^\w\s\-]")

    pre_tokenizer = Sequence([
        WhitespaceSplit(),
        Split(pattern=custom_split_pattern, behavior='isolated')
    ])

    batch_size = 1000
    token_count = Counter()

    for text in tqdm(dataset["text"]):
        tokens = pre_tokenizer.pre_tokenize_str(normalizer.normalize_str(text))
        for token in tokens:
            token_count[token[0]] += 1

    for text in tqdm(dataset["title"]):
        tokens = pre_tokenizer.pre_tokenize_str(normalizer.normalize_str(text))
        for token in tokens:
            token_count[token[0]] += 1

    new_tokens = []
    for token, freq in token_count.most_common(1000):
        if token not in base_vocab:
            new_tokens.append(token)

    print(len(new_tokens), new_tokens)

    tokenizer.add_tokens(new_tokens)
    origin_embeddings = model.get_input_embeddings().weight.clone()
    origin_bias = model.get_output_embeddings().bias.clone()

    mean_emb = origin_embeddings.data.mean(dim=0)
    mean_bias = origin_bias.data.mean()

    model.resize_token_embeddings(len(tokenizer))

    model.get_input_embeddings().weight.data[origin_embeddings.shape[0]:] = mean_emb
    model.get_output_embeddings().weight.data[origin_bias.shape[0]:] = mean_emb
    model.get_output_embeddings().bias.data[origin_bias.shape[0]:] = mean_bias

    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
