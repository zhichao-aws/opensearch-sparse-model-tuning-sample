import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
from accelerate import Accelerator
from beir.retrieval.evaluation import EvaluateRetrieval
from datasets import load_dataset

# Reuse existing components
from scripts.args import beir_datasets as default_beir_datasets
from scripts.args import nano_beir_datasets as default_nano_beir_datasets
from scripts.dataset.dataset import BEIRCorpusDataset, HFDatasetWrapper
from scripts.ingest_dense import ingest_dense
from scripts.search_dense import search_dense
from scripts.utils import emit_metrics

# Import loaders from evaluate_beir if available, otherwise define them
try:
    from evaluate_beir import load_beir_from_hf, load_nano_beir_from_hf
except ImportError:
    # Fallback implementation if import fails
    def load_beir_from_hf(
        dataset_name: str = "nfcorpus",
        split: str = "test",
        load_corpus: bool = True,
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
        if load_corpus:
            ds_corpus = load_dataset(
                f"BEIR/{dataset_name}", "corpus", split="corpus", trust_remote_code=True
            )
        else:
            ds_corpus = None
        ds_queries = load_dataset(
            f"BEIR/{dataset_name}", "queries", split="queries", trust_remote_code=True
        )
        ds_qrels = load_dataset(
            f"BEIR/{dataset_name}-qrels", split=split, trust_remote_code=True
        )

        corpus = {}
        if load_corpus:
            for r in ds_corpus:
                corpus[str(r["_id"])] = {"title": r["title"], "text": r["text"]}

        queries = {}
        for r in ds_queries:
            queries[str(r["_id"])] = r["text"]

        qrels = {}
        for r in ds_qrels:
            qid = str(r["query-id"])
            doc_id = str(r["corpus-id"])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = r["score"]

        queries = {qid: query for qid, query in queries.items() if qid in qrels}
        return corpus, queries, qrels

    def load_nano_beir_from_hf(
        dataset_name: str = "nfcorpus",
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
        ds_corpus = load_dataset(
            f"zeta-alpha-ai/{dataset_name}",
            "corpus",
            split="train",
            trust_remote_code=True,
        )
        ds_queries = load_dataset(
            f"zeta-alpha-ai/{dataset_name}",
            "queries",
            split="train",
            trust_remote_code=True,
        )
        ds_qrels = load_dataset(
            f"zeta-alpha-ai/{dataset_name}",
            "qrels",
            split="train",
            trust_remote_code=True,
        )

        corpus = {}
        for r in ds_corpus:
            corpus[str(r["_id"])] = {"title": "", "text": r["text"]}

        queries = {}
        for r in ds_queries:
            queries[str(r["_id"])] = r["text"]

        qrels = {}
        for r in ds_qrels:
            qid = str(r["query-id"])
            doc_id = str(r["corpus-id"])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][doc_id] = 1

        queries = {qid: query for qid, query in queries.items() if qid in qrels}
        return corpus, queries, qrels


logger = logging.getLogger(__name__)


def set_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "eval_beir_dense.log")),
        ],
    )


def evaluate_beir_dense(args, accelerator):
    beir_eval_dir = os.path.join(args.output_dir, "beir_eval_dense")
    os.makedirs(beir_eval_dir, exist_ok=True)

    datasets = args.beir_datasets.split(",")
    result = {
        "dataset": datasets,
        "NDCG@10": [],
        "Recall@100": [],
    }
    avg_res = dict()

    for dataset in datasets:
        dataset_name = dataset.strip()
        if not dataset_name:
            continue

        index_name = f"{dataset_name.lower()}_dense"

        # Load data
        # Only load corpus on main process or if we need to ingest?
        # Ingest needs corpus on all processes (for DDP).
        # But load_dataset might download stuff.
        # dataset library handles caching, so it's fine to call on all processes.

        logger.info(f"Loading {dataset_name}...")
        _, queries, qrels = load_beir_from_hf(
            dataset_name=dataset_name, split="test", load_corpus=False
        )

        corpus = HFDatasetWrapper(
            load_dataset(
                f"BEIR/{dataset_name}", "corpus", split="corpus", trust_remote_code=True
            ),
            sample_function=lambda x: (x["_id"], x["title"] + " " + x["text"]),
        )

        if accelerator.is_local_main_process:
            logger.info(
                f"Loaded {dataset_name} with {len(corpus)} documents and {len(queries)} queries"
            )

        if not args.skip_ingest:
            asyncio.run(
                ingest_dense(
                    dataset=corpus,
                    model_id=args.model_name_or_path,
                    index_name=index_name,
                    accelerator=accelerator,
                    max_length=args.max_seq_length,
                    batch_size=args.batch_size,
                )
            )

        # search is only run on main process
        if args.do_search and accelerator.is_local_main_process:
            search_result = asyncio.run(
                search_dense(
                    queries=queries,
                    model_id=args.model_name_or_path,
                    index_name=index_name,
                    out_dir=beir_eval_dir,
                    device=accelerator.device,
                    batch_size=args.batch_size,
                    result_size=100,  # BEIR usually evals @10, @100
                    return_text=False,
                )
            )

            ndcg, map_, recall, p = EvaluateRetrieval.evaluate(
                qrels, search_result["run_res"], [1, 10, 100]
            )
            logger.info(
                f"retrieve metrics for {dataset_name}: NDCG@10={ndcg.get('NDCG@10', 0.0)}, Recall@100={recall.get('Recall@100', 0.0)}"
            )
            result["NDCG@10"].append(ndcg.get("NDCG@10", 0.0))
            result["Recall@100"].append(recall.get("Recall@100", 0.0))

        accelerator.wait_for_everyone()

    if args.do_search and accelerator.is_local_main_process:
        df = pd.DataFrame(result)
        avg_res = {
            key: sum(result[key]) / len(result[key])
            for key in ["NDCG@10", "Recall@100"]
        }

        df.to_csv(os.path.join(beir_eval_dir, "beir_statistics.csv"))
        with open(os.path.join(beir_eval_dir, "avg_res.json"), "w") as f:
            json.dump(avg_res, f)

        doc_id = args.output_dir + "_dense"
        timestamp = datetime.now().timestamp()

        metrics = {
            "NDCG@10": avg_res["NDCG@10"],
            "Recall@100": avg_res["Recall@100"],
            "timestamp": timestamp,
            "dataset_number": len(datasets),
        }
        emit_metrics(metrics, "beir_eval_dense", doc_id)

        metrics = {
            "records": df.to_dict(orient="records"),
            "timestamp": timestamp,
        }
        emit_metrics(metrics, "beir_eval_records_dense", doc_id)

        logger.info(f"BEIR Evaluation complete. Avg NDCG@10: {avg_res['NDCG@10']}")


def evaluate_nano_beir_dense(args, accelerator):
    nano_beir_eval_dir = os.path.join(args.output_dir, "nano_beir_eval_dense")
    os.makedirs(nano_beir_eval_dir, exist_ok=True)

    datasets = args.nano_beir_datasets.split(",")
    result = {
        "dataset": datasets,
        "NDCG@10": [],
    }
    avg_res = dict()

    for dataset in datasets:
        dataset_name = dataset.strip()
        if not dataset_name:
            continue

        index_name = f"{dataset_name.lower()}_dense"

        corpus, queries, qrels = load_nano_beir_from_hf(dataset_name=dataset_name)

        if accelerator.is_local_main_process:
            logger.info(
                f"Loaded {dataset_name} with {len(corpus)} documents and {len(queries)} queries"
            )

        if not args.skip_ingest:
            asyncio.run(
                ingest_dense(
                    dataset=BEIRCorpusDataset(corpus=corpus),
                    model_id=args.model_name_or_path,
                    index_name=index_name,
                    accelerator=accelerator,
                    max_length=args.max_seq_length,
                    batch_size=args.batch_size,
                )
            )

        if args.do_search and accelerator.is_local_main_process:
            search_result = asyncio.run(
                search_dense(
                    queries=queries,
                    model_id=args.model_name_or_path,
                    index_name=index_name,
                    out_dir=nano_beir_eval_dir,
                    device=accelerator.device,
                    batch_size=args.batch_size,
                    result_size=10,
                    return_text=False,
                )
            )
            ndcg, map_, recall, p = EvaluateRetrieval.evaluate(
                qrels, search_result["run_res"], [1, 10]
            )
            logger.info(f"retrieve metrics for {dataset_name}: {ndcg, map_, recall, p}")
            result["NDCG@10"].append(ndcg.get("NDCG@10", 0.0))

        accelerator.wait_for_everyone()

    if args.do_search and accelerator.is_local_main_process:
        df = pd.DataFrame(result)

        avg_res = {key: sum(result[key]) / len(result[key]) for key in ["NDCG@10"]}

        df.to_csv(
            os.path.join(
                nano_beir_eval_dir,
                "nano_beir_statistics.csv",
            )
        )
        with open(
            os.path.join(
                nano_beir_eval_dir,
                "avg_res.json",
            ),
            "w",
        ) as f:
            json.dump(avg_res, f)

        doc_id = args.output_dir + "_dense_nano"
        timestamp = datetime.now().timestamp()

        metrics = {
            "NDCG@10": avg_res["NDCG@10"],
            "timestamp": timestamp,
            "dataset_number": len(datasets),
        }
        emit_metrics(metrics, "nano_beir_eval_dense", doc_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace model ID or path",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output_beir_dense", help="Output directory"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Max sequence length"
    )
    parser.add_argument("--skip_ingest", action="store_true", help="Skip ingestion")
    parser.add_argument("--do_search", action="store_true", help="Run search")
    parser.add_argument(
        "--beir_datasets",
        type=str,
        default=default_beir_datasets,
        help="Comma separated BEIR datasets",
    )
    parser.add_argument(
        "--nano_beir_datasets",
        type=str,
        default=default_nano_beir_datasets,
        help="Comma separated NanoBEIR datasets",
    )
    parser.add_argument("--eval_nano", action="store_true", help="Evaluate on NanoBEIR")
    parser.add_argument(
        "--no_eval_beir", action="store_true", help="Skip BEIR evaluation"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_logging(args.output_dir)

    accelerator = Accelerator()
    logger.info(
        f"Using device: {accelerator.device}, Process: {accelerator.process_index}/{accelerator.num_processes}"
    )

    if not args.no_eval_beir:
        evaluate_beir_dense(args, accelerator)

    if args.eval_nano:
        evaluate_nano_beir_dense(args, accelerator)


if __name__ == "__main__":
    main()
