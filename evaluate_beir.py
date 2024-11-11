import torch.utils
import torch.utils.data
import yaml
import logging
import os
import json
import random
import sys
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional
from tqdm import tqdm

import pandas as pd
import asyncio

from scripts.model.sparse_encoders import (
    SparseModel,
    SparseEncoder,
    sparse_embedding_to_query,
)
from scripts.data.dataset import BEIRCorpusDataset, KeyValueDataset
from scripts.ingest import ingest
from scripts.search import search
from scripts.utils import get_os_client, batch_search, set_logging, get_model
from scripts.args import ModelArguments, DataTrainingArguments, parse_args

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from accelerate import Accelerator
from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

logger = logging.getLogger(__name__)


def main():
    model_args, data_args, training_args = parse_args()

    args_dict = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
    }
    beir_eval_dir = os.path.join(training_args.output_dir, "beir_eval")
    os.makedirs(beir_eval_dir, exist_ok=True)
    with open(
        os.path.join(training_args.output_dir, "beir_eval", "config.yaml"), "w"
    ) as file:
        yaml.dump(args_dict, file, sort_keys=False)

    set_logging(training_args, "eval_beir.log")
    set_seed(training_args.seed)

    model = get_model(model_args)

    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.wait_for_everyone()

    datasets = data_args.beir_datasets.split(",")
    result = {
        "dataset": datasets,
        "flops": [],
        "NDCG@10": [],
        "q_length": [],
        "d_length": [],
    }
    avg_res = dict()
    for dataset in datasets:
        # if the dataset wasn't download before, only download it on main process
        if accelerator.is_local_main_process and not os.path.exists(
            os.path.join(data_args.beir_dir, dataset)
        ):
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
            data_path = util.download_and_unzip(url, data_args.beir_dir)
        accelerator.wait_for_everyone()
        data_path = os.path.join(data_args.beir_dir, dataset)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
            split="test"
        )

        asyncio.run(
            ingest(
                dataset=BEIRCorpusDataset(corpus=corpus),
                model=model,
                out_dir=beir_eval_dir,
                index_name=dataset,
                accelerator=accelerator,
                max_length=data_args.max_seq_length,
                batch_size=training_args.per_device_eval_batch_size,
            )
        )

        # search is only run on main process
        if accelerator.is_local_main_process:
            search_result = asyncio.run(
                search(
                    queries=queries,
                    model=model,
                    out_dir=beir_eval_dir,
                    index_name=dataset,
                    max_length=data_args.max_seq_length,
                    batch_size=training_args.per_device_eval_batch_size,
                    inf_free=model_args.inf_free,
                )
            )

            ndcg, map_, recall, p = EvaluateRetrieval.evaluate(
                qrels, search_result["run_res"], [1, 10]
            )
            logger.info(f"retrieve metrics for {dataset}: {ndcg, map_, recall, p}")
            result["NDCG@10"].append(ndcg["NDCG@10"])
            result["flops"].append(search_result["flops"])
            result["q_length"].append(search_result["q_length"])
            result["d_length"].append(search_result["d_length"])

        accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        df = pd.DataFrame(result)
        for key in ["flops", "q_length", "d_length", "NDCG@10"]:
            avg_res[key] = sum(result[key]) / len(result[key])
        df.to_csv(os.path.join(beir_eval_dir, "beir_statictics.csv"))
        with open(os.path.join(beir_eval_dir, "avg_res.json"), "w") as f:
            json.dump(avg_res, f)


if __name__ == "__main__":
    main()
