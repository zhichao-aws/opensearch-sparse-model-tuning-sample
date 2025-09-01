import asyncio
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
import yaml
from accelerate import Accelerator
from beir.retrieval.evaluation import EvaluateRetrieval
from datasets import load_dataset
from transformers import (
    set_seed,
)

from scripts.args import nano_beir_datasets, parse_args
from scripts.dataset.data_utils import cached
from scripts.dataset.dataset import BEIRCorpusDataset
from scripts.ingest import ingest
from scripts.search import search
from scripts.utils import emit_metrics, get_model, set_logging

logger = logging.getLogger(__name__)


def prepare_model_args(model_args, output_dir, step):
    model_args.model_name_or_path = os.path.join(output_dir, f"checkpoint-{step}")
    model_args.tokenizer_name = model_args.model_name_or_path
    if model_args.idf_requires_grad:
        model_args.idf_path = os.path.join(model_args.model_name_or_path, "idf.json")
    return model_args


def get_suffix(model_args, data_args):
    suffix = "_2p" if data_args.use_two_phase else ""
    suffix = suffix + (f"_{data_args.query_prune}" if data_args.query_prune > 0 else "")
    suffix = suffix + (
        f"_{data_args.eval_max_seq_length}"
        if data_args.eval_max_seq_length != 512
        else ""
    )
    suffix = suffix + (
        f"_{model_args.prune_ratio}" if model_args.prune_ratio is not None else ""
    )
    return suffix


def load_beir_from_hf(
    dataset_name: str = "nfcorpus",
    split: str = "test",
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    ds_corpus = load_dataset(
        f"BEIR/{dataset_name}", "corpus", split="corpus", trust_remote_code=True
    )
    ds_queries = load_dataset(
        f"BEIR/{dataset_name}", "queries", split="queries", trust_remote_code=True
    )
    ds_qrels = load_dataset(
        f"BEIR/{dataset_name}-qrels", split=split, trust_remote_code=True
    )

    # Build BEIR-style corpus
    corpus: Dict[str, Dict[str, str]] = {}
    for r in ds_corpus:
        corpus[str(r["_id"])] = {"title": r["title"], "text": r["text"]}

    # Build BEIR-style queries
    queries: Dict[str, str] = {}
    for r in ds_queries:
        queries[str(r["_id"])] = r["text"]

    # Build BEIR-style qrels
    qrels: Dict[str, Dict[str, int]] = {}
    for r in ds_qrels:
        qid = str(r["query-id"])
        doc_id = str(r["corpus-id"])
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = r["score"]

    # Filter queries that have qrels
    queries = {qid: query for qid, query in queries.items() if qid in qrels}
    return corpus, queries, qrels


@cached
def load_nano_beir_from_hf(
    dataset_name: str = "nfcorpus",
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    ds_corpus = load_dataset(
        f"zeta-alpha-ai/{dataset_name}", "corpus", split="train", trust_remote_code=True
    )
    ds_queries = load_dataset(
        f"zeta-alpha-ai/{dataset_name}",
        "queries",
        split="train",
        trust_remote_code=True,
    )
    ds_qrels = load_dataset(
        f"zeta-alpha-ai/{dataset_name}", "qrels", split="train", trust_remote_code=True
    )

    # Build BEIR-style corpus
    corpus: Dict[str, Dict[str, str]] = {}
    for r in ds_corpus:
        corpus[str(r["_id"])] = {"title": "", "text": r["text"]}

    # Build BEIR-style queries
    queries: Dict[str, str] = {}
    for r in ds_queries:
        queries[str(r["_id"])] = r["text"]

    # Build BEIR-style qrels
    qrels: Dict[str, Dict[str, int]] = {}
    for r in ds_qrels:
        qid = str(r["query-id"])
        doc_id = str(r["corpus-id"])
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = 1

    # Filter queries that have qrels
    queries = {qid: query for qid, query in queries.items() if qid in qrels}
    return corpus, queries, qrels


def warmup_nano_beir():
    for dataset in nano_beir_datasets.split(","):
        load_nano_beir_from_hf(dataset_name=dataset)


def evaluate_beir(model_args, data_args, training_args, model, accelerator):
    suffix = get_suffix(model_args, data_args)
    beir_eval_dir = os.path.join(training_args.output_dir, f"beir_eval{suffix}")
    os.makedirs(beir_eval_dir, exist_ok=True)

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
        corpus, queries, qrels = load_beir_from_hf(dataset_name=dataset, split="test")
        logger.info(
            f"Loaded {dataset} with {len(corpus)} documents and {len(queries)} queries"
        )
        if not data_args.skip_ingest:
            asyncio.run(
                ingest(
                    dataset=BEIRCorpusDataset(corpus=corpus),
                    model=model,
                    out_dir=beir_eval_dir,
                    index_name=dataset,
                    accelerator=accelerator,
                    max_length=data_args.eval_max_seq_length,
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
                    max_length=data_args.eval_max_seq_length,
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
        avg_res = {
            key: sum(result[key]) / len(result[key])
            for key in ["flops", "q_length", "d_length", "NDCG@10"]
        }

        df.to_csv(os.path.join(beir_eval_dir, "beir_statictics.csv"))
        with open(os.path.join(beir_eval_dir, "avg_res.json"), "w") as f:
            json.dump(avg_res, f)

        doc_id = training_args.output_dir + suffix
        timestamp = datetime.now().timestamp()

        metrics = {
            "flops": avg_res["flops"],
            "NDCG@10": avg_res["NDCG@10"],
            "q_length": avg_res["q_length"],
            "d_length": avg_res["d_length"],
            "timestamp": timestamp,
            "dataset_number": len(datasets),
        }
        emit_metrics(metrics, "beir_eval", doc_id)

        metrics = {
            "records": df.to_dict(orient="records"),
            "timestamp": timestamp,
        }
        emit_metrics(metrics, "beir_eval_records", doc_id)


def evaluate_nano_beir(model_args, data_args, training_args, model, accelerator, step):
    suffix = get_suffix(model_args, data_args)
    nano_beir_eval_dir = os.path.join(
        training_args.output_dir, f"nano_beir_eval{suffix}"
    )
    os.makedirs(nano_beir_eval_dir, exist_ok=True)

    result = {
        "dataset": nano_beir_datasets.split(","),
        "flops": [],
        "NDCG@10": [],
        "q_length": [],
        "d_length": [],
    }
    avg_res = dict()
    for dataset in nano_beir_datasets.split(","):
        corpus, queries, qrels = load_nano_beir_from_hf(dataset_name=dataset)
        logger.info(
            f"Loaded {dataset} with {len(corpus)} documents and {len(queries)} queries"
        )
        if not data_args.skip_ingest:
            asyncio.run(
                ingest(
                    dataset=BEIRCorpusDataset(corpus=corpus),
                    model=model,
                    out_dir=nano_beir_eval_dir,
                    index_name=f"{dataset}".lower(),
                    accelerator=accelerator,
                    max_length=data_args.eval_max_seq_length,
                    batch_size=training_args.per_device_eval_batch_size,
                )
            )
        if data_args.do_search and accelerator.is_local_main_process:
            search_result = asyncio.run(
                search(
                    queries=queries,
                    model=model,
                    out_dir=nano_beir_eval_dir,
                    index_name=f"{dataset}".lower(),
                    max_length=data_args.eval_max_seq_length,
                    batch_size=training_args.per_device_eval_batch_size,
                    inf_free=model_args.inf_free,
                    use_two_phase=data_args.use_two_phase,
                    query_prune=data_args.query_prune,
                )
            )
            ndcg, map_, recall, p = EvaluateRetrieval.evaluate(
                qrels, search_result["run_res"], [1, 10]
            )
            logger.info(
                f"retrieve metrics for {f'{dataset}'.lower()}: {ndcg, map_, recall, p}"
            )
            result["NDCG@10"].append(ndcg["NDCG@10"])
            result["flops"].append(search_result["flops"])
            result["q_length"].append(search_result["q_length"])
            result["d_length"].append(search_result["d_length"])

        accelerator.wait_for_everyone()

    if data_args.do_search and accelerator.is_local_main_process:
        df = pd.DataFrame(result)

        avg_res = {
            key: sum(result[key]) / len(result[key])
            for key in ["flops", "q_length", "d_length", "NDCG@10"]
        }

        df.to_csv(
            os.path.join(
                nano_beir_eval_dir,
                f"nano_beir_statictics_step{step}.csv",
            )
        )
        with open(
            os.path.join(
                nano_beir_eval_dir,
                f"avg_res_step{step}.json",
            ),
            "w",
        ) as f:
            json.dump(avg_res, f)

        doc_id = training_args.output_dir + suffix + f"_step{step}"
        timestamp = datetime.now().timestamp()

        metrics = {
            "flops": avg_res["flops"],
            "NDCG@10": avg_res["NDCG@10"],
            "q_length": avg_res["q_length"],
            "d_length": avg_res["d_length"],
            "timestamp": timestamp,
            "dataset_number": len(nano_beir_datasets.split(",")),
        }
        emit_metrics(metrics, "nano_beir_eval", doc_id)

        metrics = {
            "records": df.to_dict(orient="records"),
            "timestamp": timestamp,
        }
        emit_metrics(metrics, "nano_beir_eval_records", doc_id)


def main():
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        use_yaml = True
    else:
        use_yaml = False

    model_args, data_args, training_args = parse_args()
    if use_yaml:
        model_args = prepare_model_args(
            model_args, training_args.output_dir, training_args.max_steps
        )

    args_dict = {
        "model_args": asdict(model_args),
        "data_args": asdict(data_args),
        "training_args": asdict(training_args),
    }
    suffix = get_suffix(model_args, data_args)
    with open(
        os.path.join(training_args.output_dir, f"beir_eval_config{suffix}.yaml"), "w"
    ) as file:
        yaml.dump(args_dict, file, sort_keys=False)

    set_logging(training_args, "eval_beir.log")
    set_seed(training_args.seed)

    model = get_model(model_args)
    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.wait_for_everyone()

    evaluate_beir(model_args, data_args, training_args, model, accelerator)
    if accelerator.is_local_main_process:
        warmup_nano_beir()
    accelerator.wait_for_everyone()
    for file in os.listdir(training_args.output_dir):
        if not file.startswith("checkpoint-"):
            continue
        step = file.split("-")[-1]
        model_args.model_name_or_path = os.path.join(training_args.output_dir, file)
        model_args.tokenizer_name = model_args.model_name_or_path
        if model_args.idf_requires_grad:
            model_args.idf_path = os.path.join(
                model_args.model_name_or_path, "idf.json"
            )
        model = get_model(model_args)
        evaluate_nano_beir(
            model_args, data_args, training_args, model, accelerator, step
        )


if __name__ == "__main__":
    main()
