import argparse
import logging
import os
from typing import Dict, List, Optional, Tuple

import entmax
import torch
import utils
from fastdist import fastdist
from tokenizers import Tokenizer as RawTokenizer
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

# Ensure deepfocus is available or imported correctly
try:
    from deepfocus.focus import get_overlapping_tokens
except ImportError:
    # Assuming it's in the python path
    from deepfocus.focus import get_overlapping_tokens

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transform a source model to use a target model's vocabulary."
    )
    parser.add_argument(
        "--source_model",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Source model ID",
    )
    parser.add_argument(
        "--target_model", type=str, default="bert-base-uncased", help="Target model ID"
    )
    parser.add_argument(
        "--target_tokenizer_from_file",
        action="store_true",
        help="If set, treat --target_model as a tokenizer.json path and load it with tokenizers.Tokenizer.from_file().",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="modernbert-with-bert-vocab",
        help="Path to save the new model",
    )
    parser.add_argument(
        "--save_additional_tokens",
        action="store_true",
        help="If set, write save_path/additional_tokens.json containing all non-overlap token ids (list[int]).",
    )
    parser.add_argument(
        "--fuzzy", action="store_true", help="Enable fuzzy matching for tokens"
    )
    parser.add_argument(
        "--target_is_bpe",
        action="store_true",
        help="Set if target tokenizer is BPE and needs decoder fix",
    )
    parser.add_argument(
        "--set_bias",
        action="store_true",
        default=False,
        help="Compute and set output embedding bias",
    )
    parser.add_argument(
        "--rescale_norm",
        action="store_true",
        default=False,
        help="Rescale the norm of new embeddings to expected norm",
    )
    parser.add_argument(
        "--no_target_norm",
        action="store_false",
        dest="use_target_norm",
        help="Disable rescaling final embeddings and bias to match target model's average norm (default: enabled)",
    )
    parser.set_defaults(use_target_norm=True)
    parser.add_argument(
        "--target_norm_value",
        type=float,
        default=None,
        help="Optional. If provided, overrides the target norm used by target norm rescaling (enabled by default). "
        "When set, embeddings (and bias if present) will be rescaled so that the average norm "
        "of overlap-token embeddings equals this value, without needing to compute it from target_model.",
    )
    parser.add_argument(
        "--sim_metric",
        type=str,
        default="cosine",
        choices=["cosine", "pmi", "zscore"],
        help="Similarity metric to use",
    )
    parser.add_argument(
        "--use_matmul",
        action="store_true",
        default=False,
        help="Use matrix multiplication instead of cosine similarity",
    )
    parser.add_argument(
        "--use_mean",
        action="store_true",
        default=False,
        help="Use mean pooling of overlapped tokens for new embeddings",
    )
    parser.add_argument(
        "--use_sub",
        action="store_true",
        default=False,
        help="For non-overlapping tokens, encode token text with source tokenizer and set embedding to the mean of its subtokens' embeddings.",
    )
    parser.add_argument(
        "--all_random",
        action="store_true",
        help="Randomly initialize all token embeddings",
    )
    parser.add_argument(
        "--new_random",
        action="store_true",
        help="Randomly initialize new (non-overlapping) token embeddings",
    )
    return parser.parse_args()


def load_tokenizers(
    source_id: str,
    target_id: str,
    target_is_bpe: bool,
    target_tokenizer_from_file: bool,
) -> Tuple[PreTrainedTokenizer, PreTrainedTokenizer]:
    logger.info(
        f"Loading tokenizers: source={source_id}, target={target_id} (from_file={target_tokenizer_from_file})"
    )
    source_tokenizer = AutoTokenizer.from_pretrained(source_id)
    if target_tokenizer_from_file:
        if not os.path.isfile(target_id):
            raise FileNotFoundError(
                f"--target_tokenizer_from_file is set but file not found: {target_id}"
            )
        raw_tok = RawTokenizer.from_file(target_id)
        target_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw_tok,
            mask_token="[MASK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            do_lower_case=True,
            model_max_length=512,
        )
    else:
        target_tokenizer = AutoTokenizer.from_pretrained(target_id)

    if (
        target_is_bpe
        and getattr(target_tokenizer.backend_tokenizer, "decoder", None) is None
    ):
        if getattr(source_tokenizer.backend_tokenizer, "decoder", None) is not None:
            target_tokenizer.backend_tokenizer.decoder = (
                source_tokenizer.backend_tokenizer.decoder
            )
            logger.info(
                f"Reset target tokenizer decoder: {target_tokenizer.backend_tokenizer.decoder}"
            )
        else:
            logger.warning(
                "Target is BPE but source decoder is also None. Cannot reset decoder."
            )

    return source_tokenizer, target_tokenizer


def get_token_overlaps(
    source_tokenizer: PreTrainedTokenizer,
    target_tokenizer: PreTrainedTokenizer,
    fuzzy: bool,
) -> Tuple[List[str], List[str]]:
    logger.info("Computing overlapping tokens...")
    overlap, _ = get_overlapping_tokens(
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
        # value.source is a list of TokenMatch objects, taking the first one's native_form
        source_overlap_tokens.append(value.source[0].native_form)

    logger.info(f"Found {len(target_overlap_tokens)} overlapping tokens.")
    return target_overlap_tokens, source_overlap_tokens


def create_new_embeddings(
    source_model: PreTrainedModel,
    target_model: Optional[PreTrainedModel],
    source_tokenizer: PreTrainedTokenizer,
    target_tokenizer: PreTrainedTokenizer,
    source_overlap_tokens: List[str],
    target_overlap_tokens: List[str],
    set_bias: bool,
    rescale_norm: bool,
    sim_metric: str,
    use_matmul: bool,
    use_mean: bool,
    use_sub: bool,
    all_random: bool,
    new_random: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[int]]:
    logger.info("Creating new embedding matrix...")

    # Original embeddings
    source_emb = source_model.get_input_embeddings()
    orig_weight = source_emb.weight.data  # (V_source, dim)

    # Target vocab size (we can get this from tokenizer even if target_model isn't loaded)
    V_target = len(target_tokenizer)
    dim = orig_weight.shape[1]

    # Overlap tokens (strings) and their IDs
    overlap_ids_source = [
        source_tokenizer.convert_tokens_to_ids(tok) for tok in source_overlap_tokens
    ]
    overlap_ids_target = [
        target_tokenizer.convert_tokens_to_ids(tok) for tok in target_overlap_tokens
    ]

    # Create new embedding matrix
    new_weight = torch.zeros(
        (V_target, orig_weight.shape[1]),
        dtype=orig_weight.dtype,
        device=orig_weight.device,
    )

    # Bias handling
    new_bias = None
    if set_bias:
        logger.info("Preparing bias calculation...")
        if target_model is None:
            raise ValueError(
                "set_bias=True requires target_model to be loaded, but it is not. Please remove --set_bias or ensure target_model is loaded."
            )
        source_output = source_model.get_output_embeddings()
        target_output = target_model.get_output_embeddings()

        if (
            source_output is None
            or not hasattr(source_output, "bias")
            or source_output.bias is None
        ):
            logger.warning(
                "Source model output embeddings do not have bias. set_bias ignored."
            )
            set_bias = False
        elif (
            target_output is None
            or not hasattr(target_output, "bias")
            or target_output.bias is None
        ):
            logger.warning(
                "Target model output embeddings do not have bias. Bias calculation requires target bias."
            )
            # We can still try to use source bias, but z-score mapping won't work as intended?
            # For now, let's assume we need both.
            logger.warning("Skipping bias calculation.")
            set_bias = False
        else:
            orig_bias = source_output.bias.data
            target_output_bias = target_output.bias.data

            # Pre-calculate stats for Z-score mapping
            mu_tgt = target_output_bias.mean()
            std_tgt = target_output_bias.std()
            mu_src = orig_bias.mean()
            std_src = orig_bias.std()

            # Compute new bias for all tokens directly using target bias mapped to source distribution
            logger.info(
                "Computing new bias using Z-score mapping from target to source distribution..."
            )
            new_bias = (target_output_bias - mu_tgt) / (
                std_tgt + 1e-8
            ) * std_src + mu_src

    # Determine initializer range
    std_init = 0.02
    if hasattr(source_model.config, "initializer_range"):
        std_init = source_model.config.initializer_range
    elif hasattr(source_model.config, "init_std"):
        std_init = source_model.config.init_std

    if all_random:
        logger.info(f"Initializing ALL embeddings randomly (std={std_init})...")
        new_weight.normal_(mean=0.0, std=std_init)
        return new_weight, new_bias, list(range(V_target))

    # 1. Copy original embeddings for overlap tokens
    # Ensure indices are tensors/lists properly
    torch_tensor_overlap_source = orig_weight[overlap_ids_source]  # (|O|, dim)
    new_weight[overlap_ids_target] = torch_tensor_overlap_source

    # 2. Compute embeddings for non-overlap tokens
    all_target_ids = list(range(V_target))
    overlap_ids_target_set = set(overlap_ids_target)
    new_ids = [i for i in all_target_ids if i not in overlap_ids_target_set]

    if not new_ids:
        logger.info("All tokens overlap. No interpolation needed.")
        return new_weight, new_bias, []

    if new_random:
        logger.info(
            f"Initializing {len(new_ids)} new embeddings randomly (std={std_init})..."
        )
        # We need to assign to the specific rows.
        # Generating a smaller tensor and assigning it
        random_embeddings = torch.zeros(
            (len(new_ids), dim), dtype=new_weight.dtype, device=new_weight.device
        )
        random_embeddings.normal_(mean=0.0, std=std_init)
        new_weight[new_ids] = random_embeddings
        return new_weight, new_bias, new_ids

    logger.info(f"Computing embeddings for {len(new_ids)} non-overlapping tokens...")

    # Prepare tensors for calculation
    source_overlap = torch_tensor_overlap_source  # (|O|, dim) in source space

    if use_mean:
        logger.info("Using mean pooling of overlapping tokens for new embeddings...")
        mean_source_overlap = source_overlap.mean(dim=0)  # (dim,)
        new_rows = mean_source_overlap.repeat(len(new_ids), 1)  # (N_new, dim)

        # We skip rescale_norm logic that depends on 'weights' if use_mean is active,
        # unless we want to adapt it. For now, following transform_mean.py which just uses mean.
        if rescale_norm:
            logger.warning("rescale_norm is ignored when use_mean is True.")
    elif use_sub:
        logger.info(
            "Using source-tokenizer subtoken averaging for new embeddings (use_sub=True)..."
        )
        if rescale_norm:
            logger.warning("rescale_norm is ignored when use_sub is True.")

        # Detect whether target tokenizer is WordPiece (BERT-style). We only need a best-effort
        # heuristic to decide whether to prefix a leading space for non-"##" tokens.
        is_target_wordpiece = False
        try:
            backend_tok = getattr(target_tokenizer, "backend_tokenizer", None)
            backend_model = getattr(backend_tok, "model", None)
            model_name = getattr(backend_model, "__class__", type("x", (), {})).__name__
            is_target_wordpiece = model_name.lower() == "wordpiece"
        except Exception:
            is_target_wordpiece = False

        # Heuristic normalization: target vocab tokens are often "token strings" not raw text.
        # We map common tokenizer markers into something source_tokenizer.encode() can handle.
        def _token_to_text(tok: str) -> str:
            # WordPiece continuation marker
            if tok.startswith("##") and len(tok) > 2:
                return tok[2:]
            # WordPiece "beginning-of-word": prefix a space so source tokenizer is more likely
            # to treat it as a word start. Skip special tokens like [CLS]/[SEP].
            if is_target_wordpiece and tok and (not tok.startswith("##")):
                if tok.startswith("[") and tok.endswith("]"):
                    return tok
                return " " + tok
            # RoBERTa/GPT2 BPE "beginning-of-word with space" marker
            if tok.startswith("Ġ") and len(tok) > 1:
                return " " + tok[1:]
            # SentencePiece "beginning-of-word with space" marker
            if tok.startswith("▁") and len(tok) > 1:
                return " " + tok[1:]
            return tok

        flat_sub_ids: List[int] = []
        flat_row_ids: List[int] = []
        counts = torch.zeros(
            len(new_ids), device=orig_weight.device, dtype=torch.float32
        )

        unk_id = getattr(source_tokenizer, "unk_token_id", None)

        for row_idx, tgt_id in enumerate(new_ids):
            tok = target_tokenizer.convert_ids_to_tokens(int(tgt_id))
            text = _token_to_text(tok)
            sub_ids = source_tokenizer.encode(text, add_special_tokens=False)
            # print(tok, source_tokenizer.convert_ids_to_tokens(sub_ids))

            if not sub_ids:
                if unk_id is not None:
                    sub_ids = [int(unk_id)]
                else:
                    # As a last resort, just use the first embedding row.
                    sub_ids = [0]

            # If encode collapsed to UNK only, keep it (still a deterministic fallback)
            for sid in sub_ids:
                flat_sub_ids.append(int(sid))
                flat_row_ids.append(row_idx)
            counts[row_idx] = float(len(sub_ids))

        sub_ids_t = torch.tensor(
            flat_sub_ids, device=orig_weight.device, dtype=torch.long
        )
        row_ids_t = torch.tensor(
            flat_row_ids, device=orig_weight.device, dtype=torch.long
        )
        sub_embs = orig_weight[sub_ids_t]  # (N_flat, dim)

        sums = torch.zeros(
            (len(new_ids), dim),
            device=orig_weight.device,
            dtype=orig_weight.dtype,
        )
        sums.index_add_(0, row_ids_t, sub_embs)

        # Avoid division by zero (counts should be >0 due to fallbacks)
        new_rows = sums / (counts.unsqueeze(1).to(sums.dtype) + 1e-8)
    else:
        if target_model is None:
            raise ValueError(
                "Similarity interpolation with target_model token vectors is required, but target_model is not loaded. Please remove interpolation settings other than --use_mean/--all_random/--new_random, or ensure target_model is loaded."
            )
        # Target embeddings needed for similarity-based interpolation
        target_emb = target_model.get_input_embeddings()
        target_weight = target_emb.weight.data  # (V_target, dim)
        if target_weight.shape[0] != V_target:
            logger.warning(
                f"Tokenizer vocab size ({V_target}) != target embedding rows ({target_weight.shape[0]}). Using target embedding rows."
            )
            V_target = target_weight.shape[0]

        target_overlap = target_weight[overlap_ids_target]  # (|O|, dim) in target space
        target_new = target_weight[new_ids]  # (N_new, dim)
        # Calculate similarity
        logger.info(
            f"Calculating similarity using {sim_metric} (matmul={use_matmul})..."
        )

        if use_matmul:
            # Dot product similarity
            sims_tensor = torch.matmul(target_new, target_overlap.T)  # (N_new, |O|)
        else:
            # Cosine similarity
            # fastdist is fast on CPU, but if we want to do complex transforms (pmi/zscore)
            # on the full matrix, we might want it as a tensor.
            # Let's stick to torch for flexibility if not using simple cosine

            # Note: fastdist returns numpy.
            target_new_np = target_new.cpu().numpy()
            target_overlap_np = target_overlap.cpu().numpy()

            sims_np = fastdist.cosine_matrix_to_matrix(
                target_new_np, target_overlap_np
            )  # (N_new, |O|)
            sims_tensor = torch.from_numpy(sims_np).to(orig_weight.device).float()

        # Apply metric transformations if needed
        if sim_metric == "pmi":
            # PMI expects non-negative input usually, or prob distributions.
            # If we used matmul (unnormalized), it might need normalization first?
            # utils.pmi_similarity expects X
            sims_tensor = utils.pmi_similarity(sims_tensor)
        elif sim_metric == "zscore":
            sims_tensor = utils.zscore_column_similarity(sims_tensor)

        # Use sparsemax for sparse attention weights
        weights = entmax.sparsemax(sims_tensor).to(torch.float32)  # (N_new, |O|)
        logger.info(
            f"avg nonzero elements in weights: {weights.nonzero().shape[0] / weights.shape[0]}"
        )

        # Weighted sum in Source space
        new_rows = torch.matmul(weights, source_overlap)  # (N_new, dim)

        if rescale_norm:
            logger.info("Rescaling norms of new embeddings...")
            # source_overlap is (|O|, dim)
            # weights is (N_new, |O|)

            overlap_norms = source_overlap.norm(dim=1)  # (|O|,)
            expected_norms = torch.matmul(weights, overlap_norms)  # (N_new,)

            current_norms = new_rows.norm(dim=1)  # (N_new,)

            logger.info(
                f"expected_norms.mean(): {expected_norms.mean()}, current_norms.mean(): {current_norms.mean()}, source_overlap.norm(dim=1).mean(): {source_overlap.norm(dim=1).mean()}"
            )
            logger.info(f"expected_norms: {expected_norms}")
            logger.info(f"current_norms: {current_norms}")

            # Avoid division by zero
            scale_factor = expected_norms / (current_norms + 1e-8)
            new_rows = new_rows * scale_factor.unsqueeze(1)

    # Assign to new_weight
    new_weight[new_ids] = new_rows

    return new_weight, new_bias, new_ids


def apply_and_save(
    source_model: PreTrainedModel,
    target_model: Optional[PreTrainedModel],
    target_tokenizer: PreTrainedTokenizer,
    overlap_ids_target: List[int],
    new_weight: torch.Tensor,
    new_bias: Optional[torch.Tensor],
    additional_token_ids: List[int],
    save_path: str,
    use_target_norm: bool,
    target_norm_value: Optional[float],
    save_additional_tokens: bool,
):
    logger.info("Applying new embeddings to model...")

    if use_target_norm or (target_norm_value is not None):
        logger.info(
            "Rescaling final embeddings and bias to match target norm (on overlap tokens)..."
        )

        # 1. Decide the target norm to match
        if target_norm_value is not None:
            avg_target_norm = torch.tensor(
                float(target_norm_value),
                device=new_weight.device,
                dtype=new_weight.dtype,
            )
            logger.info(f"Using provided target_norm_value={float(target_norm_value)}")
        else:
            if target_model is None:
                raise ValueError(
                    "use_target_norm=True requires loading target_model to calculate target norm, but it is not currently loaded. "
                    "Please provide --target_norm_value, or ensure target_model is loaded."
                )
            # Calculate average norm of overlap tokens in TARGET model
            target_emb = target_model.get_input_embeddings()
            target_weight = target_emb.weight.data
            target_overlap_norms = target_weight[overlap_ids_target].norm(dim=1)
            avg_target_norm = target_overlap_norms.mean()

        # 2. Calculate average norm of overlap tokens in NEW embeddings (which are from Source)
        # new_weight already has overlap tokens set to source values
        new_overlap_norms = new_weight[overlap_ids_target].norm(dim=1)
        avg_new_norm = new_overlap_norms.mean()

        scale_factor_emb = avg_target_norm / (avg_new_norm + 1e-8)
        logger.info(
            f"Embedding Scale Factor: {scale_factor_emb:.4f} (Target Avg Norm: {avg_target_norm:.4f} / Source Avg Norm: {avg_new_norm:.4f})"
        )

        new_weight = new_weight * scale_factor_emb

        if new_bias is not None:
            # Bias should be scaled consistently with embeddings.
            new_bias = new_bias * scale_factor_emb

    # Create new embedding layer
    new_emb_layer = torch.nn.Embedding.from_pretrained(new_weight, freeze=False)

    # Resize and set embeddings
    source_model.resize_token_embeddings(len(target_tokenizer))
    source_model.set_input_embeddings(new_emb_layer)

    if new_bias is not None:
        logger.info("Setting output embeddings with bias...")
    else:
        new_bias = torch.zeros(
            new_weight.shape[0], device=new_weight.device, dtype=new_weight.dtype
        )

    vocab_size, hidden_dim = new_weight.shape
    new_decoder = torch.nn.Linear(hidden_dim, vocab_size, bias=True)

    # Tie weights manually if needed, or let the model handle it?
    # Ideally: new_decoder.weight = new_emb_layer.weight
    # But new_decoder.weight is Parameter, new_emb_layer.weight is Parameter.
    new_decoder.weight = new_emb_layer.weight
    new_decoder.bias.data = new_bias

    source_model.set_output_embeddings(new_decoder)

    # Update config with special tokens from target tokenizer
    # We map standard BERT special tokens to the config
    special_tokens_map = {
        "bos_token_id": "[CLS]",
        "eos_token_id": "[SEP]",
        "pad_token_id": "[PAD]",
        "sep_token_id": "[SEP]",
        "cls_token_id": "[CLS]",
    }

    # HF tokenizers differ: slow tokenizers usually expose .vocab, fast tokenizers prefer .get_vocab()
    target_vocab: Dict[str, int]
    if hasattr(target_tokenizer, "get_vocab"):
        target_vocab = target_tokenizer.get_vocab()
    else:
        target_vocab = getattr(target_tokenizer, "vocab", {})

    for config_key, token_text in special_tokens_map.items():
        if token_text in target_vocab:
            setattr(source_model.config, config_key, target_vocab[token_text])
        else:
            # Try conversion if not in vocab dict directly
            token_id = target_tokenizer.convert_tokens_to_ids(token_text)
            if token_id != target_tokenizer.unk_token_id:
                setattr(source_model.config, config_key, token_id)
            else:
                logger.warning(
                    f"Token {token_text} not found in target vocab, skipping {config_key}"
                )

    logger.info(f"Saving model to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    source_model.save_pretrained(save_path)
    target_tokenizer.save_pretrained(save_path)

    if save_additional_tokens:
        import json

        out_path = os.path.join(save_path, "additional_tokens.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(additional_token_ids, f)
        logger.info(
            f"Wrote additional token ids to {out_path} (count={len(additional_token_ids)})"
        )
    logger.info("Done.")


def main():
    args = parse_args()

    # Validate mutually exclusive modes for generating NEW (non-overlap) embeddings
    new_modes = [args.use_mean, args.use_sub, args.all_random, args.new_random]
    if sum(bool(x) for x in new_modes) > 1:
        raise ValueError(
            "Parameter conflict: only one of --use_mean/--use_sub/--all_random/--new_random can be enabled at a time."
        )

    # Load Tokenizers
    source_tokenizer, target_tokenizer = load_tokenizers(
        args.source_model,
        args.target_model,
        args.target_is_bpe,
        args.target_tokenizer_from_file,
    )

    # Get Overlaps
    target_overlap_tokens, source_overlap_tokens = get_token_overlaps(
        source_tokenizer, target_tokenizer, args.fuzzy
    )

    # Load Models
    # target_model is only needed when target embedding/bias/norm is required
    needs_target_model = (
        (args.use_target_norm and args.target_norm_value is None)
        or args.set_bias
        or (
            (not args.use_mean)
            and (not args.use_sub)
            and (not args.all_random)
            and (not args.new_random)
        )
    )

    logger.info(f"Loading source model: {args.source_model}")
    source_model = AutoModelForMaskedLM.from_pretrained(
        args.source_model, trust_remote_code=True
    )

    target_model = None
    if needs_target_model:
        logger.info(f"Loading target model: {args.target_model}")
        target_model = AutoModelForMaskedLM.from_pretrained(
            args.target_model, trust_remote_code=True
        )
    else:
        logger.info("Skipping target model load (not needed for current settings).")

    # Compute New Embeddings
    new_weight, new_bias, additional_token_ids = create_new_embeddings(
        source_model,
        target_model,
        source_tokenizer,
        target_tokenizer,
        source_overlap_tokens,
        target_overlap_tokens,
        args.set_bias,
        args.rescale_norm,
        args.sim_metric,
        args.use_matmul,
        args.use_mean,
        args.use_sub,
        args.all_random,
        args.new_random,
    )

    # Need overlap IDs for rescaling logic in apply_and_save
    overlap_ids_target = [
        target_tokenizer.convert_tokens_to_ids(tok) for tok in target_overlap_tokens
    ]

    # Apply and Save
    apply_and_save(
        source_model,
        target_model,
        target_tokenizer,
        overlap_ids_target,
        new_weight,
        new_bias,
        additional_token_ids,
        args.save_path,
        args.use_target_norm,
        args.target_norm_value,
        args.save_additional_tokens,
    )


if __name__ == "__main__":
    main()