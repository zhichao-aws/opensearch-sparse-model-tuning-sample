from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    ModernBertConfig,
    ModernBertForMaskedLM,
    ModernBertModel,
    ModernBertPreTrainedModel,
)
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertPredictionHead,
    _pad_modernbert_output,
    _unpad_modernbert_input,
)


class AlignmentMDBertConfig(ModernBertConfig):
    uncased_vocab_size = 30522
    model_type = "alignment-modernbert"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_vocab_size = kwargs.get(
            "target_vocab_size", self.uncased_vocab_size
        )
        self.tie_word_embeddings = False


class AlignmentMDBertForMaskedLM(ModernBertForMaskedLM):
    config_class = AlignmentMDBertConfig

    def __init__(self, config: AlignmentMDBertConfig):
        ModernBertPreTrainedModel.__init__(self, config)
        self.config = config
        self.model = ModernBertModel(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(
            config.hidden_size, config.target_vocab_size, bias=config.decoder_bias
        )

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

        # Initialize weights and apply final processing
        self.post_init()

    # NOTE:
    # - We intentionally keep input embeddings on `config.vocab_size` (base tokenizer).
    # - But logits are produced with `config.target_vocab_size` (target tokenizer).
    #   ModernBertForMaskedLM.forward computes the masked-LM loss using `config.vocab_size`,
    #   which breaks when output vocab != input vocab. We override forward and compute the
    #   loss using `logits.shape[-1]` (i.e., target vocab).
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], MaskedLMOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        self._maybe_set_compile()

        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                if batch_size is None and seq_len is None:
                    if inputs_embeds is not None:
                        batch_size, seq_len = inputs_embeds.shape[:2]
                    else:
                        batch_size, seq_len = input_ids.shape[:2]
                device = (
                    input_ids.device if input_ids is not None else inputs_embeds.device
                )

                if attention_mask is None:
                    attention_mask = torch.ones(
                        (batch_size, seq_len), device=device, dtype=torch.bool
                    )

                if inputs_embeds is None:
                    with torch.no_grad():
                        (
                            input_ids,
                            indices,
                            cu_seqlens,
                            max_seqlen,
                            position_ids,
                            labels,
                        ) = _unpad_modernbert_input(
                            inputs=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            labels=labels,
                        )
                else:
                    (
                        inputs_embeds,
                        indices,
                        cu_seqlens,
                        max_seqlen,
                        position_ids,
                        labels,
                    ) = _unpad_modernbert_input(
                        inputs=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        labels=labels,
                    )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        if self.sparse_prediction and labels is not None:
            # flatten labels and output first
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

            # then filter out the non-masked tokens
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

        logits = (
            self.compiled_head(last_hidden_state)
            if self.config.reference_compile
            else self.decoder(self.head(last_hidden_state))
        )

        loss = None
        if labels is not None:
            # IMPORTANT: output vocab size may differ from input vocab size
            loss = self.loss_function(
                logits, labels, vocab_size=int(logits.shape[-1]), **kwargs
            )

        if self.config._attn_implementation == "flash_attention_2":
            with (
                nullcontext()
                if self.config.repad_logits_with_grad or labels is None
                else torch.no_grad()
            ):
                logits = _pad_modernbert_output(
                    inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len
                )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


AutoConfig.register("alignment-modernbert", AlignmentMDBertConfig)
AutoModelForMaskedLM.register(AlignmentMDBertConfig, AlignmentMDBertForMaskedLM)
