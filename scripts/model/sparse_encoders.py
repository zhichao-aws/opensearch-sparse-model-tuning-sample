import torch
import transformers
import itertools
import logging
from transformers import T5EncoderModel, T5Tokenizer, AutoConfig, AutoModelForMaskedLM

from .decomposition_bert import DecompBertConfig, DecompBertForMaskedLM

logger = logging.getLogger(__name__)


class SparseModel(torch.nn.Module):
    def __init__(
        self,
        model_id,
        idf=None,
        tokenizer_id=None,
        split_batch=1,
        idf_requires_grad=False,
        model_type="bert",
    ):
        super().__init__()

        self.model_type = model_type
        if tokenizer_id is None:
            tokenizer_id = model_id

        if model_type == "flan-t5":
            self.backbone = T5EncoderModel.from_pretrained(model_id)
            self.tokenizer = T5Tokenizer.from_pretrained(
                tokenizer_id
            )  # actullay we can also use the bert tokenizer as the vocab size
            self.vocab_projection = torch.nn.Linear(
                self.backbone.config.d_model, len(self.tokenizer)
            )
            self.special_token_ids = []
            for special_token in self.tokenizer.all_special_tokens:
                token_id = self.tokenizer.convert_tokens_to_ids(special_token)
                self.special_token_ids.append(token_id)
        else:
            AutoConfig.register("DecompBert", DecompBertConfig)
            AutoModelForMaskedLM.register(DecompBertConfig, DecompBertForMaskedLM)
            self.backbone = AutoModelForMaskedLM.from_pretrained(model_id)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_id)
            self.special_token_ids = [
                self.tokenizer.vocab[token]
                for token in self.tokenizer.special_tokens_map.values()
            ]

        vocab_size = len(self.tokenizer)
        idf_vector = [1.0] * vocab_size
        if idf is not None:
            logger.info(f"set idf to the model. requires_grad: {idf_requires_grad}")
            for token, weight in idf.items():
                if self.model_type == "flan-t5":
                    _id = self.tokenizer.convert_tokens_to_ids(token)
                else:
                    _id = self.tokenizer._convert_token_to_id_with_added_voc(token)
                idf_vector[_id] = weight

        self.idf_vector = torch.nn.Parameter(
            torch.tensor(idf_vector), requires_grad=idf_requires_grad
        )
        self.split_batch = split_batch
        self.idf_requires_grad = idf_requires_grad

    def forward(self, inf_free=False, **kwargs):
        if inf_free:
            return self._encode_inf_free(**kwargs)
        else:
            return self._encode(**kwargs)

    def _encode(self, **kwargs):
        if self.model_type == "flan-t5":
            outputs = self.backbone(
                input_ids=kwargs["input_ids"],
                attention_mask=kwargs.get("attention_mask"),
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            logits = self.vocab_projection(hidden_states)
            values, _ = torch.max(
                logits * kwargs.get("attention_mask").unsqueeze(-1), dim=1
            )
            return torch.log(1 + torch.relu(values))

        if self.split_batch == 1:
            output = self.backbone(**kwargs)[0]
            values, _ = torch.max(
                output * kwargs.get("attention_mask").unsqueeze(-1), dim=1
            )
            return torch.log(1 + torch.relu(values))

        batch_size = kwargs["input_ids"].size(0)
        split_sizes = [batch_size // self.split_batch] * self.split_batch
        remainder = batch_size % self.split_batch
        for i in range(remainder):
            split_sizes[i] += 1

        outputs = []
        start = 0
        attention_mask = kwargs.get("attention_mask")
        output = self.backbone.bert(**kwargs)[0]
        output = self.backbone.cls.predictions.transform(output)
        for split_size in split_sizes:
            end = start + split_size
            values = self.backbone.cls.predictions.decoder(output[start:end])
            values, _ = torch.max(
                values * attention_mask[start:end].unsqueeze(-1), dim=1
            )
            outputs.append(values)
            start = end

        output = torch.cat(outputs, dim=0)
        return torch.log(1 + torch.relu(output))

    def _encode_inf_free(self, **kwargs):
        input_ids = kwargs.get("input_ids")
        batch_size = input_ids.shape[0]
        out = torch.zeros(batch_size, len(self.tokenizer), device=input_ids.device)
        out[torch.arange(batch_size).unsqueeze(-1), input_ids] = 1
        out[:, self.special_token_ids] = 0
        return out * torch.relu(self.idf_vector)


class SparsePostProcessor(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        if isinstance(tokenizer, T5Tokenizer):
            self.id_to_token = tokenizer.convert_ids_to_tokens(range(len(tokenizer)))
        else:
            self.id_to_token = ["" for i in range(tokenizer.vocab_size)]
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
        self.device = next(self.model.parameters()).device
        self.count_tensor = torch.zeros(len(self.tokenizer)).to(self.device)

    def reset_count(self):
        self.count_tensor = torch.zeros(len(self.tokenizer)).to(self.device)

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


def sparse_embedding_to_query(token_weight_map, field_name="text_sparse"):
    clause_list = []
    for token, weight in token_weight_map.items():
        clause_list.append(
            {
                "rank_feature": {
                    "field": f"{field_name}.{token}",
                    "boost": weight,
                    "linear": {},
                }
            }
        )
    return {"bool": {"should": clause_list}}
