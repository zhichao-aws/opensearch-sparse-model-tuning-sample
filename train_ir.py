import logging
import os
import sys
from dataclasses import asdict

import yaml
from torch.optim import AdamW
from transformers import set_seed
from transformers.optimization import get_linear_schedule_with_warmup

from scripts.args import parse_args
from scripts.async_embedding_server import EmbeddingService
from scripts.dataset.collator import (
    COLLATOR_CLS_MAP,
)
from scripts.dataset.dataset import (
    load_dataset,
    load_datasets,
)
from scripts.train.loss import LOSS_CLS_MAP
from scripts.train.trainer import SparseModelTrainer
from scripts.utils import get_model, set_logging

logger = logging.getLogger(__name__)


def main():
    model_args, data_args, training_args = parse_args()
    os.makedirs(training_args.output_dir, exist_ok=True)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        os.system(
            f"""cp {sys.argv[1]} {os.path.join(training_args.output_dir, "train_config.yaml")}"""
        )
    else:
        args_dict = {
            "model_args": asdict(model_args),
            "data_args": asdict(data_args),
            "training_args": asdict(training_args),
        }
        with open(os.path.join(training_args.output_dir, "config.yaml"), "w") as file:
            yaml.dump(args_dict, file, sort_keys=False)

    set_logging(training_args, "train.log")
    set_seed(training_args.seed)

    # set up embedding server
    embedding_service = None
    if (
        len(data_args.kd_ensemble_teacher_kwargs) != 0
        and "remote" in data_args.kd_ensemble_teacher_kwargs["types"]
    ):
        embedding_service = EmbeddingService()
        logger.info(embedding_service.health_check())
        logger.info("embedding service has been started.")

    # model
    model = get_model(model_args)

    # data collator
    data_collator = COLLATOR_CLS_MAP[data_args.data_type](
        model.tokenizer,
        data_args.max_seq_length,
        data_args.kd_ensemble_teacher_kwargs.get("teacher_tokenizer_ids", []),
        embedding_service=embedding_service,
    )
    logger.info(f"data collator: {data_collator}")

    # loss functions
    loss_functions = []
    for loss_type in data_args.loss_types:
        loss_cls = LOSS_CLS_MAP[loss_type]
        logger.info(f"add loss: {loss_cls}")
        loss_functions.append(
            loss_cls(
                use_in_batch_negatives=data_args.use_in_batch_negatives,
                weight=data_args.ranking_loss_weight,
                temperature=data_args.temperature,
            )
        )

    # optimizer
    if not model_args.idf_requires_grad or data_args.idf_lr is None:
        optimizer = AdamW(
            model.parameters(),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
        )
    else:
        idf_vector_params = [model.idf_vector]
        other_params = [
            param for param in model.parameters() if param is not model.idf_vector
        ]
        param_groups = [
            {"params": idf_vector_params, "lr": data_args.idf_lr},
            {"params": other_params, "lr": training_args.learning_rate},
        ]
        optimizer = AdamW(param_groups, weight_decay=training_args.weight_decay)
        logger.info(f"idf_vector lr: {data_args.idf_lr}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=training_args.max_steps,
    )

    if data_args.train_file is not None:
        dataset = load_dataset(
            path=data_args.train_file,
            cls=data_args.data_type,
            swap_times=data_args.swap_times,
            sample_num_one_query=data_args.sample_num_one_query,
            first_rank_thresh=data_args.first_rank_thresh,
        )
    elif data_args.train_file_dir is not None:
        dataset = load_datasets(
            path=data_args.train_file_dir,
            cls=data_args.data_type,
            training_args=training_args,
            swap_times=data_args.swap_times,
            sample_num_one_query=data_args.sample_num_one_query,
            first_rank_thresh=data_args.first_rank_thresh,
        )
    else:
        raise ValueError("train_file or train_file_dir must be specified")

    trainer = SparseModelTrainer(
        model_args=model_args,
        data_args=data_args,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        loss_functions=loss_functions,
        optimizers=(optimizer, scheduler),
    )

    if len(data_args.kd_ensemble_teacher_kwargs) != 0:
        logger.info(f"Set bi-encoder teacher. {data_args.kd_ensemble_teacher_kwargs}")
        trainer.set_bi_encoder_teacher(embedding_service=embedding_service)
    trainer.train()


if __name__ == "__main__":
    try:
        main()
    finally:
        os.system("pkill -f async_embedding_server")
