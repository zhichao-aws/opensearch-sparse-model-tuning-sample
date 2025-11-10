from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import logging

from scripts.model.models import *

logger = logging.get_logger(__name__)


class SAPTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.teacher = AutoModelForMaskedLM.from_pretrained("splade-v3")
        self.teacher.eval()
        self._move_model_to_device(self.teacher, self.args.device)
        self.teacher = self._wrap_model(self.teacher, training=False)
        self.accelerator.prepare(self.teacher)

        self.loss_fn = nn.MSELoss(reduction="mean")

    def splade_encode(self, model, input_ids, attention_mask):
        output = model(input_ids=input_ids, attention_mask=attention_mask)[0]
        values, _ = torch.max(output * attention_mask.unsqueeze(-1), dim=1)
        return values

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        teacher_input_ids = inputs["teacher_input_ids"]
        teacher_attention_mask = inputs["teacher_attention_mask"]

        logits = self.splade_encode(model, input_ids, attention_mask)
        teacher_logits = self.splade_encode(
            self.teacher, teacher_input_ids, teacher_attention_mask
        )

        reps_mask = (logits > 0) | (teacher_logits > 0)
        loss = self.loss_fn(logits[reps_mask], teacher_logits[reps_mask])
        return loss


class SAPDataCollator:
    def __init__(self, model_tokenizer, teacher_tokenizer):
        self.model_tokenizer = model_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

    def __call__(self, examples):
        texts = [ex["text"] for ex in examples]

        model_batch = self.model_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )

        teacher_batch = self.teacher_tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )

        return {
            "input_ids": model_batch["input_ids"],
            "attention_mask": model_batch["attention_mask"],
            "teacher_input_ids": teacher_batch["input_ids"],
            "teacher_attention_mask": teacher_batch["attention_mask"],
        }


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization."},
    )
    train_only_embeddings: bool = field(
        default=False,
        metadata={
            "help": "If true, only parameters with names containing 'decoder' or 'head' are trainable.",
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    model = AlignmentMDBertForMaskedLM.from_pretrained(model_args.model_name_or_path)
    set_seed(training_args.seed)

    dataset = load_dataset(
        "json", data_files="data/wikibook.ml128.jsonl", streaming=False
    )["train"]

    # Keep original dataset columns (e.g., 'text') for custom data collator
    training_args.remove_unused_columns = False

    model_tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=True
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained("splade-v3", use_fast=True)
    data_collator = SAPDataCollator(
        model_tokenizer=model_tokenizer, teacher_tokenizer=teacher_tokenizer
    )

    if model_args.train_only_embeddings:
        for name, param in model.named_parameters():
            allow_train = ("decoder" in name) or ("head" in name)
            param.requires_grad = allow_train

    trainer = SAPTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    trainer.train()


if __name__ == "__main__":
    main()
