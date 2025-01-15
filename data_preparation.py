# data_preparation.py
import os
import json
import asyncio
import pandas as pd
from datasets import load_dataset
from transformers import HfArgumentParser
from accelerate import Accelerator

from scripts.ingest import ingest
from scripts.search import search
from scripts.utils import get_model
from scripts.args import ModelArguments, DataTrainingArguments
from tqdm import tqdm

from torch.utils.data import Dataset


class MSMarcoCorpusDataset(Dataset):
    def __init__(self, corpus):
        self.corpus = corpus
        self.ids = list(corpus.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        doc_id = self.ids[idx]
        # Return a tuple of (id, text) as expected by the ingest function
        return doc_id, self.corpus[doc_id]["text"]


def load_msmarco_data(split="train", is_main_process=True):
    if is_main_process:
        print(f"Loading MS MARCO {split} split...")
    dataset = load_dataset("ms_marco", "v2.1")

    corpus = {}
    queries = {}
    qrels = {}

    iterator = tqdm(
        dataset[split], desc="Processing dataset", disable=not is_main_process
    )

    for item in iterator:
        query_id = str(item["query_id"])
        queries[query_id] = item["query"]

        if query_id not in qrels:
            qrels[query_id] = {}

        for idx, (passage_text, is_selected) in enumerate(
            zip(item["passages"]["passage_text"], item["passages"]["is_selected"])
        ):
            doc_id = f"{query_id}_{idx}"
            corpus[doc_id] = {"text": passage_text, "title": ""}

            if is_selected:
                qrels[query_id][doc_id] = 1

    return corpus, queries, qrels


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Setup accelerator
    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.wait_for_everyone()

    # Get model using the reference method
    model = get_model(model_args)

    # Load MS MARCO dataset
    corpus, queries, qrels = load_msmarco_data(
        split="train", is_main_process=accelerator.is_main_process
    )

    # Load IDF values if specified
    if model_args.idf_path and os.path.exists(model_args.idf_path):
        with open(model_args.idf_path, "r") as f:
            idf_values = json.load(f)
        model.set_idf_values(idf_values)

    # Ingest corpus into OpenSearch
    asyncio.run(
        ingest(
            dataset=MSMarcoCorpusDataset(corpus=corpus),
            model=model,
            out_dir="tmp/out",
            index_name="msmarco",
            accelerator=accelerator,
            max_length=data_args.max_seq_length,
            batch_size=50,
        )
    )

    # Perform search for negative sampling
    if accelerator.is_main_process:
        search_result = asyncio.run(
            search(
                queries=queries,
                model=model,
                out_dir="tmp/out",
                index_name="msmarco",
                max_length=data_args.max_seq_length,
                batch_size=50,
                result_size=50,
            )
        )

        # Prepare training data
        data = []
        run_res = search_result["run_res"]
        for query, docs in run_res.items():
            if query not in qrels:
                continue
            # Remove positive documents from negative candidates
            for doc_id in qrels[query]:
                if doc_id in docs:
                    docs.pop(doc_id)

            # Create training examples
            for positive in list(qrels[query].keys()):
                data.append(
                    {
                        "query": queries[query],
                        "pos": corpus[positive]["text"],
                        "negs": [corpus[neg]["text"] for neg in list(docs.keys())],
                    }
                )

        # Save as HuggingFace dataset
        from datasets import Dataset

        ds = Dataset.from_list(data)
        ds.save_to_disk("data/msmarco_train")


if __name__ == "__main__":
    main()
