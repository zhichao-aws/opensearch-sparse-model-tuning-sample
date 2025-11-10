import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    ModernBertConfig,
    ModernBertForMaskedLM,
    ModernBertModel,
    ModernBertPreTrainedModel,
)
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead


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


AutoConfig.register("alignment-modernbert", AlignmentMDBertConfig)
AutoModelForMaskedLM.register(AlignmentMDBertConfig, AlignmentMDBertForMaskedLM)
