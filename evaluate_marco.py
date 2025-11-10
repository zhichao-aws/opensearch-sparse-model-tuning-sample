import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict

import ir_datasets
from accelerate import Accelerator
from beir.retrieval.evaluation import EvaluateRetrieval
from datasets import load_dataset
from transformers import (
    set_seed,
)

from evaluate_beir import get_suffix, load_beir_from_hf, prepare_model_args
from scripts.args import parse_args
from scripts.dataset.dataset import HFDatasetWrapper, MsmarcoAccessor
from scripts.ingest import ingest
from scripts.search import search
from scripts.utils import emit_metrics, get_model, set_logging

logger = logging.getLogger(__name__)


def load_trec_dl(year):
    ds = ir_datasets.load(f"msmarco-passage/trec-dl-{year}")
    queries = {q.query_id: q.text for q in ds.queries_iter()}

    qrels = defaultdict(dict)
    for qr in ds.qrels_iter():
        qrels[qr.query_id][qr.doc_id] = qr.relevance  # relevance is graded (0..3)

    queries = {qid: query for qid, query in queries.items() if qid in qrels}
    return queries, qrels


def _sorted_docids_by_score(run_doc_scores: Dict[str, float]):
    return [
        doc_id
        for doc_id, _ in sorted(
            run_doc_scores.items(), key=lambda x: x[1], reverse=True
        )
    ]


def compute_mrr_at_k(
    run_res: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]], k: int = 10
) -> float:
    mrr_total = 0.0
    evaluated = 0
    for qid, doc_scores in run_res.items():
        if qid not in qrels:
            continue
        ranked_docids = _sorted_docids_by_score(doc_scores)[:k]
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(ranked_docids, start=1):
            if qrels[qid].get(doc_id, 0) > 0:
                reciprocal_rank = 1.0 / rank
                break
        mrr_total += reciprocal_rank
        evaluated += 1
    return mrr_total / evaluated if evaluated > 0 else 0.0


def evaluate_msmarco_dev(model_args, data_args, training_args, model, accelerator):
    suffix = get_suffix(model_args, data_args)
    out_dir = os.path.join(training_args.output_dir, "evaluate_marco")
    os.makedirs(out_dir, exist_ok=True)

    dataset = "msmarco"
    index_name = dataset
    _, queries, qrels = load_beir_from_hf(
        dataset_name=dataset, split="validation", load_corpus=False
    )
    corpus = HFDatasetWrapper(
        load_dataset("BeIR/msmarco", "corpus", split="corpus"),
        sample_function=lambda x: (x["_id"], MsmarcoAccessor.transform_str(x["text"])),
    )

    logger.info(
        f"Loaded {dataset} dev with {len(corpus)} documents and {len(queries)} queries"
    )

    if not data_args.skip_ingest:
        asyncio.run(
            ingest(
                dataset=corpus,
                model=model,
                out_dir=out_dir,
                index_name=index_name,
                accelerator=accelerator,
                max_length=data_args.eval_max_seq_length,
                batch_size=training_args.per_device_eval_batch_size,
                tokenizer_out=model_args.tokenizer_out,
            )
        )

    metrics = {}
    if data_args.do_search and accelerator.is_local_main_process:
        search_result = asyncio.run(
            search(
                queries=queries,
                model=model,
                out_dir=out_dir,
                index_name=index_name,
                max_length=data_args.eval_max_seq_length,
                batch_size=training_args.per_device_eval_batch_size,
                inf_free=model_args.inf_free,
                use_two_phase=data_args.use_two_phase,
                query_prune=data_args.query_prune,
                result_size=1000,
                tokenizer_out=model_args.tokenizer_out,
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

        with open(os.path.join(out_dir, f"msmarco_metrics{suffix}.json"), "w") as f:
            json.dump(metrics, f)

        logger.info(f"MSMARCO dev metrics: {metrics}")

        doc_id = training_args.output_dir + suffix
        timestamp = datetime.now().timestamp()

        metrics = {
            "flops": search_result["flops"],
            "MRR@10": metrics["MRR@10"],
            "Recall@10": metrics["Recall@10"],
            "Recall@1000": metrics["Recall@1000"],
            "timestamp": timestamp,
        }
        emit_metrics(metrics, "msmarco_eval", doc_id)

    accelerator.wait_for_everyone()
    return metrics


def evaluate_trec_dl(model_args, data_args, training_args, model, accelerator):
    suffix = get_suffix(model_args, data_args)
    out_dir = os.path.join(training_args.output_dir, "evaluate_marco")
    index_name = "msmarco"

    if data_args.do_search and accelerator.is_local_main_process:
        metrics = {}
        for year in [2019, 2020]:
            queries, qrels = load_trec_dl(year)

            search_result = asyncio.run(
                search(
                    queries=queries,
                    model=model,
                    out_dir=out_dir,
                    index_name=index_name,
                    max_length=data_args.eval_max_seq_length,
                    batch_size=training_args.per_device_eval_batch_size,
                    inf_free=model_args.inf_free,
                    use_two_phase=data_args.use_two_phase,
                    query_prune=data_args.query_prune,
                    result_size=1000,
                    tokenizer_out=model_args.tokenizer_out,
                )
            )

            run_res = search_result["run_res"]
            ndcg, map_, recall, p = EvaluateRetrieval.evaluate(
                qrels, run_res, [10, 1000]
            )

            metrics[f"trec-dl-{year}-ndcg@10"] = ndcg.get("NDCG@10", 0.0)
            metrics[f"trec-dl-{year}-Recall@10"] = recall.get("Recall@10", 0.0)
            metrics[f"trec-dl-{year}-Recall@1000"] = recall.get("Recall@1000", 0.0)

        with open(os.path.join(out_dir, f"trec_dl_metrics{suffix}.json"), "w") as f:
            json.dump(metrics, f)


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

    set_logging(training_args, "eval_msmarco.log")
    set_seed(training_args.seed)

    model = get_model(model_args)
    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.prepare(model)
    accelerator.wait_for_everyone()

    evaluate_msmarco_dev(model_args, data_args, training_args, model, accelerator)
    evaluate_trec_dl(model_args, data_args, training_args, model, accelerator)


if __name__ == "__main__":
    main()
