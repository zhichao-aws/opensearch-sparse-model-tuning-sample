import itertools
import logging

import torch
import transformers

logger = logging.getLogger(__name__)


class TokenizerWithProcessing:
    def __init__(self, original, process=None):
        self._original = original
        self.process = process

    def __call__(self, text, **kwargs):
        assert isinstance(text, list)
        assert isinstance(text[0], str)
        if self.process is not None:
            text = self.process(text)
        return self._original.__call__(text, **kwargs)

    def __getattr__(self, name):
        return getattr(self._original, name)


class TextPreProcessors:
    @staticmethod
    def to_lower(texts):
        texts = [text.lower() for text in texts]
        return texts

    @staticmethod
    def blank_prefix(texts):
        texts = [" " + text for text in texts]
        return texts

    @staticmethod
    def blank_prefix_lower(texts):
        texts = [" " + text.lower() for text in texts]
        return texts


class SparseModel(torch.nn.Module):
    def __init__(
        self,
        model_id,
        idf=None,
        tokenizer_id=None,
        idf_requires_grad=False,
        prune_ratio=None,
        preprocess_func=None,
        use_l0=True,
    ):
        super().__init__()

        if tokenizer_id is None:
            tokenizer_id = model_id
        self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)
        if preprocess_func is not None:
            logger.info(f"Using preprocess function {preprocess_func}")
            func = getattr(TextPreProcessors, preprocess_func)
            sample = ["Hello WorldABC."]
            logger.info(f"before:{sample}, after:{func(sample)}")
            self.tokenizer = TokenizerWithProcessing(self.tokenizer, func)

        self.special_token_ids = [
            self.tokenizer.vocab[token]
            for token in self.tokenizer.special_tokens_map.values()
        ]

        self.vocab_size = len(self.tokenizer.vocab)
        try:
            emb_vocab_size = self.backbone.new.embeddings.word_embeddings.weight.shape[
                0
            ]
            if emb_vocab_size != self.vocab_size:
                logger.info(
                    f"reset the vocab size from {self.vocab_size} to {emb_vocab_size}"
                )
                self.vocab_size = emb_vocab_size
        except Exception:
            pass

        idf_vector = [1.0] * self.vocab_size
        if idf is not None:
            logger.info(f"set idf to the model. requires_grad: {idf_requires_grad}")
            for token, weight in idf.items():
                _id = self.tokenizer._convert_token_to_id_with_added_voc(token)
                idf_vector[_id] = weight
        self.idf_vector = torch.nn.Parameter(
            torch.tensor(idf_vector), requires_grad=idf_requires_grad
        )
        self.idf_requires_grad = idf_requires_grad
        self.prune_ratio = prune_ratio
        self.use_l0 = use_l0
        logger.info(f"model prune ratio: {self.prune_ratio}, use l0: {self.use_l0}")

    def forward(self, inf_free=False, **kwargs):
        # input kwargs is the features from tokenizer
        if inf_free:
            return self._encode_inf_free(**kwargs)
        else:
            return self._encode(**kwargs)

    def _encode(self, **kwargs):
        output = self.backbone(**kwargs)[0]
        values, _ = torch.max(
            output * kwargs.get("attention_mask").unsqueeze(-1), dim=1
        )
        values = torch.log1p(torch.relu(values))
        if self.use_l0:
            values = torch.log1p(values)
        if self.prune_ratio is None:
            return values
        else:
            max_values = values.max(dim=-1)[0].unsqueeze(1) * self.prune_ratio
            return values * (values > max_values)

    def _encode_inf_free(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        batch_size = input_ids.shape[0]
        out = torch.zeros(batch_size, self.vocab_size, device=input_ids.device)
        out[torch.arange(batch_size).unsqueeze(-1), input_ids] = 1
        out[:, self.special_token_ids] = 0
        return out * torch.relu(self.idf_vector)


class SparsePostProcessor(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.id_to_token = ["" for i in range(len(tokenizer.vocab) + 100)]
        for token, _id in tokenizer.vocab.items():
            self.id_to_token[_id] = token

    def __call__(self, sparse_vector):
        sparse_vector[:, 0] = 1
        sample_indices, token_indices = torch.nonzero(sparse_vector, as_tuple=True)
        non_zero_values = sparse_vector[(sample_indices, token_indices)].tolist()
        number_of_tokens_for_each_sample = torch.bincount(sample_indices).cpu().tolist()
        tokens = [self.id_to_token[_id] for _id in token_indices.tolist()]

        output = []
        end_idxs = list(itertools.accumulate([0] + number_of_tokens_for_each_sample))
        for i in range(len(end_idxs) - 1):
            token_strings = tokens[end_idxs[i] : end_idxs[i + 1]]
            weights = non_zero_values[end_idxs[i] : end_idxs[i + 1]]
            output.append(dict(zip(token_strings[1:], weights[1:])))
        return output


class SparseEncoder:
    def __init__(self, sparse_model, max_length, do_count=True):
        self.model = sparse_model
        self.tokenizer = sparse_model.tokenizer
        self.post_processor = SparsePostProcessor(tokenizer=sparse_model.tokenizer)
        self.do_count = do_count
        self.max_length = max_length
        self.device = self.model.backbone.device
        self.count_tensor = torch.zeros(self.model.vocab_size).to(self.device)

    def reset_count(self):
        self.count_tensor = torch.zeros(self.model.vocab_size).to(self.device)

    def encode(self, texts, inf_free=False):
        features = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=self.max_length,
        )
        features = {k: v.to(self.device) for k, v in features.items()}
        with torch.no_grad():
            output = self.model(inf_free=inf_free, **features)
        if self.do_count:
            self.count_tensor += (output > 0).long().sum(dim=0)
        output = self.post_processor(output)
        return output


def sparse_embedding_to_query(
    token_weight_map, field_name="text_sparse", query_prune=0
):
    if query_prune > 0:
        thresh = max(token_weight_map.values()) * query_prune
        token_weight_map = {
            token: weight
            for token, weight in token_weight_map.items()
            if weight > thresh
        }
    return {"neural_sparse": {field_name: {"query_tokens": token_weight_map}}}
