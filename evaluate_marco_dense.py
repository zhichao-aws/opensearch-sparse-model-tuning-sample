import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime

from accelerate import Accelerator
from beir.retrieval.evaluation import EvaluateRetrieval
from datasets import load_dataset

# Assuming running from root
from evaluate_beir import load_beir_from_hf
from evaluate_marco import compute_mrr_at_k, load_trec_dl
from scripts.dataset.dataset import HFDatasetWrapper, MsmarcoAccessor
from scripts.ingest_dense import ingest_dense
from scripts.search_dense import search_dense
from scripts.utils import emit_metrics

logger = logging.getLogger(__name__)


def set_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(output_dir, "eval_msmarco_dense.log")),
        ],
    )


def evaluate_msmarco_dev_dense(args, accelerator):
    out_dir = os.path.join(args.output_dir, "evaluate_marco_dense")
    os.makedirs(out_dir, exist_ok=True)

    dataset_name = "msmarco"
    index_name = "msmarco_dense"

    # Load queries and qrels
    _, queries, qrels = load_beir_from_hf(
        dataset_name=dataset_name, split="validation", load_corpus=False
    )

    # Load corpus for ingest
    corpus = HFDatasetWrapper(
        load_dataset("BeIR/msmarco", "corpus", split="corpus"),
        sample_function=lambda x: (x["_id"], MsmarcoAccessor.transform_str(x["text"])),
    )

    if accelerator.is_local_main_process:
        logger.info(
            f"Loaded {dataset_name} dev with {len(corpus)} documents and {len(queries)} queries"
        )

    if not args.skip_ingest:
        asyncio.run(
            ingest_dense(
                dataset=corpus,
                model_id=args.model_name_or_path,
                index_name=index_name,
                accelerator=accelerator,
                batch_size=args.batch_size,
                max_length=args.max_seq_length,
            )
        )

    metrics = {}
    # Search is run only on main process
    if args.do_search and accelerator.is_local_main_process:
        search_result = asyncio.run(
            search_dense(
                queries=queries,
                model_id=args.model_name_or_path,
                index_name=index_name,
                out_dir=out_dir,
                device=accelerator.device,
                batch_size=args.batch_size,
                result_size=1000,
                return_text=False,
            )
        )

        run_res = search_result["run_res"]
        mrr10 = compute_mrr_at_k(run_res, qrels, k=10)
        ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, run_res, [10, 1000])

        metrics = {
            "MRR@10": mrr10,
            "Recall@10": recall.get("Recall@10", 0.0),
            "Recall@1000": recall.get("Recall@1000", 0.0),
        }

        with open(os.path.join(out_dir, "msmarco_metrics_dense.json"), "w") as f:
            json.dump(metrics, f)

        logger.info(f"MSMARCO dev dense metrics: {metrics}")

        doc_id = args.output_dir + "_dense"
        timestamp = datetime.now().timestamp()

        metrics.update({"timestamp": timestamp})
        emit_metrics(metrics, "msmarco_eval_dense", doc_id)

    accelerator.wait_for_everyone()
    return metrics


def evaluate_trec_dl_dense(args, accelerator):
    out_dir = os.path.join(args.output_dir, "evaluate_marco_dense")
    os.makedirs(out_dir, exist_ok=True)
    index_name = "msmarco_dense"

    # Search is run only on main process
    if args.do_search and accelerator.is_local_main_process:
        metrics = {}
        for year in [2019, 2020]:
            queries, qrels = load_trec_dl(year)

            search_result = asyncio.run(
                search_dense(
                    queries=queries,
                    model_id=args.model_name_or_path,
                    index_name=index_name,
                    out_dir=out_dir,
                    device=accelerator.device,
                    batch_size=args.batch_size,
                    result_size=1000,
                    return_text=False,
                )
            )

            run_res = search_result["run_res"]
            ndcg, map_, recall, p = EvaluateRetrieval.evaluate(
                qrels, run_res, [10, 1000]
            )

            metrics[f"trec-dl-{year}-ndcg@10"] = ndcg.get("NDCG@10", 0.0)
            metrics[f"trec-dl-{year}-Recall@10"] = recall.get("Recall@10", 0.0)
            metrics[f"trec-dl-{year}-Recall@1000"] = recall.get("Recall@1000", 0.0)

        with open(os.path.join(out_dir, "trec_dl_metrics_dense.json"), "w") as f:
            json.dump(metrics, f)

        logger.info(f"TREC DL dense metrics: {metrics}")

    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace model ID or path",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output_dense", help="Output directory"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max_seq_length", type=int, default=512, help="Max sequence length"
    )
    parser.add_argument("--skip_ingest", action="store_true", help="Skip ingestion")
    parser.add_argument("--do_search", action="store_true", help="Run search")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_logging(args.output_dir)

    accelerator = Accelerator()
    logger.info(
        f"Using device: {accelerator.device}, Process: {accelerator.process_index}/{accelerator.num_processes}"
    )

    evaluate_msmarco_dev_dense(args, accelerator)
    evaluate_trec_dl_dense(args, accelerator)


if __name__ == "__main__":
    main()
