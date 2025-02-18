import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
from transformers import BertModel, BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertPredictionHeadTransform, BertOnlyMLMHead

from transformers import BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import *

# which is bias tied to?


class TransposeLinear(torch.nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight.T, self.bias)


class DecompBertConfig(BertConfig):
    model_type = "DecompBert"

    def __init__(self, vocab_intermediate_size=256, **kwargs):
        super().__init__(**kwargs)
        self.vocab_intermediate_size = vocab_intermediate_size


class DecompBertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.vocab_intermediate_size,
            padding_idx=config.pad_token_id,
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.word_embeddings_projection = nn.Linear(
            config.vocab_intermediate_size, config.hidden_size, bias=False
        )

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[
                :, past_key_values_length : seq_length + past_key_values_length
            ]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device
                )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            inputs_embeds = self.word_embeddings_projection(inputs_embeds)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class DecompBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = DecompBertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.attn_implementation = config._attn_implementation
        self.position_embedding_type = config.position_embedding_type

        # Initialize weights and apply final processing
        self.post_init()


class DecompBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.word_embeddings_projection = TransposeLinear(
            config.vocab_intermediate_size, config.hidden_size, bias=False
        )
        self.decoder = nn.Linear(
            config.vocab_intermediate_size, config.vocab_size, bias=False
        )

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def _tie_weights(self):
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.word_embeddings_projection(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class DecompBertOnlyMLMHead(BertOnlyMLMHead):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = DecompBertLMPredictionHead(config)


class DecompBertForMaskedLM(BertForMaskedLM):
    config_class = DecompBertConfig
    _tied_weights_keys = [
        "cls.predictions.decoder.bias",
        "cls.predictions.decoder.weight",
        "cls.predictions.word_embeddings_projection.weight",
    ]

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = DecompBertModel(config, add_pooling_layer=False)
        self.cls = DecompBertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()
        print("finish init for DecompBertForMaskedLM")

    def _tie_weights(self):
        if self.config.torchscript:
            self.cls.predictions.word_embeddings_projection.weight = nn.Parameter(
                self.bert.embeddings.word_embeddings_projection.weight.clone()
            )
        else:
            self.cls.predictions.word_embeddings_projection.weight = (
                self.bert.embeddings.word_embeddings_projection.weight
            )
