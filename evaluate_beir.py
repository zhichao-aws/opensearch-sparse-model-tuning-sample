import yaml
import logging
import os
import json
import torch
from dataclasses import asdict

import pandas as pd
import asyncio

from scripts.dataset.dataset import BEIRCorpusDataset
from scripts.ingest import ingest
from scripts.search import search
from scripts.utils import set_logging, get_model
from scripts.args import parse_args

from transformers import (
    set_seed,
)

from accelerate import Accelerator
from beir import util
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
    eval_output_dir = (
        "beir_eval"
        if model_args.prune_ratio is None
        else f"beir_eval_{model_args.prune_ratio}"
    )
    beir_eval_dir = os.path.join(training_args.output_dir, eval_output_dir)
    os.makedirs(beir_eval_dir, exist_ok=True)
    with open(os.path.join(beir_eval_dir, "config.yaml"), "w") as file:
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

        if not data_args.skip_ingest:
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
        if data_args.do_search and accelerator.is_local_main_process:
            search_result = asyncio.run(
                search(
                    queries=queries,
                    model=model,
                    out_dir=beir_eval_dir,
                    index_name=dataset,
                    max_length=data_args.max_seq_length,
                    batch_size=training_args.per_device_eval_batch_size,
                    inf_free=model_args.inf_free,
                    use_two_phase=data_args.use_two_phase,
                    query_prune=data_args.query_prune,
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

    if data_args.do_search and accelerator.is_local_main_process:
        df = pd.DataFrame(result)
        for key in ["flops", "q_length", "d_length", "NDCG@10"]:
            avg_res[key] = sum(result[key]) / len(result[key])

        suffix = "_2p" if data_args.use_two_phase else ""
        suffix = suffix + (
            f"_{data_args.query_prune}" if data_args.query_prune > 0 else ""
        )
        suffix = suffix + (
            f"_{data_args.max_seq_length}" if data_args.max_seq_length != 512 else ""
        )

        df.to_csv(
            os.path.join(
                beir_eval_dir,
                f"beir_statictics{suffix}.csv",
            )
        )
        with open(
            os.path.join(
                beir_eval_dir,
                f"avg_res{suffix}.json",
            ),
            "w",
        ) as f:
            json.dump(avg_res, f)


if __name__ == "__main__":
    main()
