import itertools
import logging

import torch
import transformers

logger = logging.getLogger(__name__)


class KnowledgeDistillDataCollator:
    def __init__(self, tokenizer, max_length=512, teacher_tokenizer_ids=[], **kwargs):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizers = [self.tokenizer] + [
            transformers.AutoTokenizer.from_pretrained(tokenizer_id)
            for tokenizer_id in teacher_tokenizer_ids
        ]
        logger.info(f"total tokenizers {len(self.tokenizers)}")
        logger.info(f"unused args: {kwargs}")

    # handle knowledge distillation input containing query, docs and scores
    def __call__(self, batch):
        q, docs, scores = zip(*batch)

        has_scores = list(scores)[0][0] is not None
        assert len(docs) == len(scores)
        docs = list(itertools.chain(*docs))

        result = {"query": [], "docs": []}

        for tokenizer in self.tokenizers:
            result["query"].append(
                tokenizer(
                    list(q),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
            )
            result["docs"].append(
                tokenizer(
                    docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
            )

        if has_scores:
            result["scores"] = torch.tensor(scores)

        return result


class KnowledgeDistillIdsDataCollator:
    def __init__(
        self,
        tokenizer,
        max_length=512,
        teacher_tokenizer_ids=[],
        embedding_service=None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizers = [self.tokenizer] + [
            (
                transformers.AutoTokenizer.from_pretrained(tokenizer_id)
                if not tokenizer_id.isdigit()
                else int(tokenizer_id)
            )
            for tokenizer_id in teacher_tokenizer_ids
        ]
        self.embedding_service = embedding_service
        logger.info(f"total tokenizers {len(self.tokenizers)}")

    # handle knowledge distillation input containing query, docs and scores
    def __call__(self, batch):
        q, q_id, docs, d_ids, scores = zip(*batch)

        has_scores = list(scores)[0][0] is not None
        assert len(docs) == len(scores)
        docs = list(itertools.chain(*docs))
        d_ids = list(itertools.chain(*d_ids))

        result = {"query": [], "docs": []}

        for tokenizer in self.tokenizers:
            if isinstance(tokenizer, int):
                q_id = list(q_id)
                d_ids = list(d_ids)

                self.embedding_service.register_task(
                    table_name="vector_q", model_id=tokenizer, ids=q_id
                )
                self.embedding_service.register_task(
                    table_name="vector", model_id=tokenizer, ids=d_ids
                )

                result["query"].append({"q_id": torch.tensor(q_id)})
                result["docs"].append({"d_ids": torch.tensor(d_ids)})
                continue

            result["query"].append(
                tokenizer(
                    list(q),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
            )
            result["docs"].append(
                tokenizer(
                    docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
            )

        if has_scores:
            result["scores"] = torch.tensor(scores)

        return result


class PosNegsDataCollator:
    def __init__(self, tokenizer, max_length=512, teacher_tokenizer_ids=[], **kwargs):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizers = [self.tokenizer] + [
            transformers.AutoTokenizer.from_pretrained(tokenizer_id)
            for tokenizer_id in teacher_tokenizer_ids
        ]
        logger.info(f"total tokenizers {len(self.tokenizers)}")
        logger.info(f"unused args: {kwargs}")

    def __call__(self, batch):
        q, pos, negs = zip(*batch)
        assert len(q) == len(pos)
        docs = []
        for p, neg in zip(pos, negs):
            docs.append(p)
            docs = docs + neg

        result = {"query": [], "docs": []}

        for tokenizer in self.tokenizers:
            result["query"].append(
                tokenizer(
                    list(q),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
            )
            result["docs"].append(
                tokenizer(
                    docs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
            )
        return result


COLLATOR_CLS_MAP = {
    "kd": KnowledgeDistillDataCollator,
    "posnegs": PosNegsDataCollator,
    "kd-ids": KnowledgeDistillIdsDataCollator,
}
