import os
import json
import asyncio

from scripts.data.dataset import BEIRCorpusDataset
from scripts.ingest import ingest
from scripts.search import search
from scripts.utils import get_model
from scripts.args import ModelArguments, DataTrainingArguments

from datasets import Dataset
from transformers import HfArgumentParser
from accelerate import Accelerator
from beir import util
from beir.datasets.data_loader import GenericDataLoader


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))

    model_args, data_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(mixed_precision="fp16")
    accelerator.wait_for_everyone()
    model = get_model(model_args)

    dataset = data_args.beir_datasets.split(",")
    if len(dataset) != 1:
        raise Exception("can only accept one beir dataset")
    else:
        dataset = dataset[0]
    if accelerator.is_local_main_process and not os.path.exists(
        os.path.join(data_args.beir_dir, dataset)
    ):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        data_path = util.download_and_unzip(url, data_args.beir_dir)
    accelerator.wait_for_everyone()
    data_path = os.path.join(data_args.beir_dir, dataset)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(
        split="train"
    )

    asyncio.run(
        ingest(
            dataset=BEIRCorpusDataset(corpus=corpus),
            model=model,
            out_dir="tmp/out",
            index_name=dataset,
            accelerator=accelerator,
            max_length=data_args.max_seq_length,
            batch_size=50,
        )
    )

    if accelerator.is_main_process:
        search_result = asyncio.run(
            search(
                queries=queries,
                model=model,
                out_dir="tmp/out",
                index_name=dataset,
                max_length=data_args.max_seq_length,
                batch_size=50,
                result_size=50,
            )
        )

        data = []
        run_res = search_result["run_res"]
        for query, docs in run_res.items():
            if query not in qrels:
                continue
            for doc_id in qrels[query]:
                if doc_id in docs:
                    docs.pop(doc_id)
            for positive in list(qrels[query].keys()):
                data.append(
                    {
                        "query": queries[query],
                        "pos": corpus[positive]["title"]
                        + " "
                        + corpus[positive]["text"],
                        "negs": [
                            corpus[neg]["title"] + " " + corpus[neg]["text"]
                            for neg in list(docs.keys())
                        ],
                    }
                )

        ds = Dataset.from_list(data)
        ds.save_to_disk(f"data/{dataset}_train")


if __name__ == "__main__":
    main()
