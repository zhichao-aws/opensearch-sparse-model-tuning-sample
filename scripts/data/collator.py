import torch
import itertools
import logging

import transformers

logger = logging.getLogger(__name__)


class KnowledgeDistillDataCollator:
    def __init__(self, tokenizer, max_length=512, teacher_tokenizer_ids=[]):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizers = [self.tokenizer] + [
            transformers.AutoTokenizer.from_pretrained(tokenizer_id)
            for tokenizer_id in teacher_tokenizer_ids
        ]
        logger.info(f"total tokenizers {len(self.tokenizers)}")

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


class PosNegsDataCollator:
    def __init__(self, tokenizer, max_length=512, teacher_tokenizer_ids=[]):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizers = [self.tokenizer] + [
            transformers.AutoTokenizer.from_pretrained(tokenizer_id)
            for tokenizer_id in teacher_tokenizer_ids
        ]
        logger.info(f"total tokenizers {len(self.tokenizers)}")

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


COLLATOR_CLS_MAP = {"kd": KnowledgeDistillDataCollator, "posnegs": PosNegsDataCollator}
