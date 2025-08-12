import os
import requests
import torch
import transformers
import numpy as np
from scripts.utils import gather_rep
import logging

logger = logging.getLogger(__name__)


class BiSparseModel(torch.nn.Module):
    @staticmethod
    def from_pretrained(path):
        return BiSparseModel(path)

    def __init__(self, model_id):
        super().__init__()
        self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.special_token_ids = [
            self.tokenizer.vocab[token]
            for token in self.tokenizer.special_tokens_map.values()
        ]

    def forward(self, **kwargs):
        output = self.backbone(**kwargs)[0]
        values, _ = torch.max(
            output * kwargs.get("attention_mask").unsqueeze(-1), dim=1
        )
        values = torch.log(1 + torch.relu(values))
        values[:, self.special_token_ids] = 0
        return values


class DenseModel(torch.nn.Module):
    @staticmethod
    def from_pretrained(path):
        return DenseModel(path)

    @staticmethod
    def get_dense_embedding(output):
        sentence_embeddings = output[0][:, 0]
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )
        return sentence_embeddings

    def __init__(self, model_id):
        super().__init__()
        self.backbone = transformers.AutoModel.from_pretrained(
            model_id, trust_remote_code=True
        )

    def forward(self, **kwargs):
        output = self.backbone(**kwargs)
        return DenseModel.get_dense_embedding(output)


class RemoteModel(torch.nn.Module):
    @staticmethod
    def from_pretrained(path):
        return RemoteModel(path)

    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id
        assert "AWS_REGION_NAME" in os.environ
        self.region_name = os.environ.get("AWS_REGION_NAME")
        self.embedding_service = None

    def forward(self, **kwargs):
        if "q_id" in kwargs:
            ids = kwargs["q_id"]
            table_name = "vector_q"
        else:
            ids = kwargs["d_ids"]
            table_name = "vector"

        emb = self.embedding_service.fetch_embedding(
            table_name=table_name,
            model_id=self.model_id,
            ids=ids.cpu().numpy().tolist(),
        )
        emb = emb.reshape(ids.shape[0], -1)
        return torch.tensor(emb).to(ids.device)


class BiEncoderWrapper:
    CLS_MAP = {"sparse": BiSparseModel, "dense": DenseModel, "remote": RemoteModel}

    def __init__(
        self,
        types,
        model_ids,
        score_scale=30,
        use_in_batch_negatives=False,
        embedding_service=None,
    ):
        assert len(types) == len(model_ids)
        assert len(types) != 0
        self.score_scale = score_scale
        self.use_in_batch_negatives = use_in_batch_negatives
        self.models = []
        self.accelerator = None

        for i, (type, model_id) in enumerate(zip(types, model_ids)):
            cls = BiEncoderWrapper.CLS_MAP[type]
            model = cls.from_pretrained(model_id)
            model.eval()
            if isinstance(model, RemoteModel):
                model.embedding_service = embedding_service
            self.models.append(model)

    def get_scores_batch(self, q_features_list, d_features_list):
        assert len(q_features_list) == len(self.models)
        scores = 0
        with torch.no_grad():
            for i, model in enumerate(self.models):
                q_rep = model(**q_features_list[i])
                d_rep = model(**d_features_list[i])
                if not self.use_in_batch_negatives:
                    d_rep = d_rep.reshape(q_rep.shape[0], -1, d_rep.shape[-1])
                    score = torch.bmm(
                        d_rep, q_rep.reshape(q_rep.shape[0], q_rep.shape[-1], 1)
                    ).squeeze()
                else:
                    d_rep = gather_rep(d_rep, self.accelerator)
                    score = torch.matmul(q_rep, d_rep.t())

                max_t = score.max(dim=1).values
                min_t = score.min(dim=1).values
                score = (score - min_t.unsqueeze(-1)) / (
                    (max_t - min_t + 1e-6).unsqueeze(-1)
                )
                scores += score
                # idx = [i for i in range(q_rep.shape[0])]
                # idx_j = [i * 3 for i in range(q_rep.shape[0])]
                # logger.info(score[idx, idx_j].mean())
                # logger.info(score.max(dim=-1).indices)

            scores = scores / len(self.models)

        return scores * self.score_scale
