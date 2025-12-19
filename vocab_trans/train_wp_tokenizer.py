import argparse
import json
from typing import Iterable, List, Optional

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, processors
from tokenizers.trainers import WordPieceTrainer


def train_bert_like_wordpiece(
    texts: Iterable[str],
    vocab_size: int = 30_000,
    min_frequency: int = 2,
    lowercase: bool = True,
    strip_accents: bool = True,
    special_tokens: Optional[List[str]] = None,
) -> Tokenizer:
    """
    Train a BERT-like WordPiece tokenizer from an in-memory list of strings
    using HuggingFace `tokenizers`.

    Notes:
      - This is "BERT-like": BertNormalizer + BertPreTokenizer + WordPiece model.
      - For BERT-uncased behavior: lowercase=True, strip_accents=True
      - For BERT-cased behavior:   lowercase=False (strip_accents usually False too)
    """
    if special_tokens is None:
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    # 1) WordPiece model (like BERT)
    tokenizer = Tokenizer(
        models.WordPiece(
            unk_token="[UNK]",
            continuing_subword_prefix="##",  # BERT-style
        )
    )

    # 2) Normalization + pre-tokenization (BERT-style)
    tokenizer.normalizer = normalizers.BertNormalizer(
        lowercase=lowercase,
        strip_accents=strip_accents,
    )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    # 3) Trainer
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Train from the in-memory iterator of strings
    length = len(texts) if hasattr(texts, "__len__") else None
    tokenizer.train_from_iterator(texts, trainer=trainer, length=length)

    # 4) Post-processing (adds [CLS]/[SEP] like BERT for single/pair sequences)
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    if cls_id is not None and sep_id is not None:
        tokenizer.post_processor = processors.BertProcessing(
            ("[SEP]", sep_id),
            ("[CLS]", cls_id),
        )

    # 5) Decoder (to stitch WordPiece back together, handling "##")
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    return tokenizer


def load_jsonl_texts(file_path: str) -> List[str]:
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "text" in data:
                    texts.append(data["text"])
            except json.JSONDecodeError:
                continue
    return texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERT-like WordPiece tokenizer.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file (must contain 'text' field).")
    parser.add_argument("--vocab_size", type=int, default=20000, help="Vocabulary size.")
    parser.add_argument("--min_frequency", type=int, default=5, help="Minimum frequency for a token to be included.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the trained tokenizer JSON.")
    parser.add_argument("--lowercase", action="store_true", default=True, help="Whether to lowercase inputs.")
    parser.add_argument("--no_lowercase", action="store_false", dest="lowercase", help="Do not lowercase inputs.")
    
    args = parser.parse_args()

    print(f"Loading texts from {args.input_file}...")
    texts = load_jsonl_texts(args.input_file)
    print(f"Loaded {len(texts)} texts.")
    
    print(f"Training tokenizer with vocab_size={args.vocab_size}, min_freq={args.min_frequency}...")
    tok = train_bert_like_wordpiece(
        texts,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        lowercase=args.lowercase,
        strip_accents=args.lowercase, # usually consistent with lowercase for BERT
    )

    print(f"Saving tokenizer to {args.save_path}...")
    tok.save(args.save_path)
    print("Done.")
