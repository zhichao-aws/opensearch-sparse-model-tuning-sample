# data_preparation.py
import os
import json
import asyncio
import numpy as np
from datasets import load_dataset
from transformers import HfArgumentParser
from accelerate import Accelerator

from scripts.ingest import ingest
from scripts.search import search
from scripts.utils import get_model
from scripts.args import ModelArguments, DataTrainingArguments
from tqdm import tqdm
import torch
import torch.distributed as dist
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import CrossEncoder
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.nn.functional as F

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

def normalize_scores(scores):
    """Normalize scores to [0,1] range within batch"""
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)


def process_batch_with_teacher(teacher, queries, documents):
    """Process a batch of query-doc pairs with a teacher model"""
    scores = teacher(queries, documents)
    return normalize_scores(scores)


def init_distributed():
    """Initialize distributed training"""
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, world_size


class GTETeacher:
    def __init__(self, local_rank):
        self.device = torch.device(f"cuda:{local_rank}")
        self.base_model = SentenceTransformer(
            "Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True
        )
        self.base_model.to(self.device)
        self.model = DDP(self.base_model, device_ids=[local_rank])

    def __call__(self, queries, documents):
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(documents, str):
            documents = [documents]

        with torch.cuda.amp.autocast():
            query_embeddings = self.base_model.encode(
                queries, convert_to_tensor=True
            ).to(self.device)
            doc_embeddings = self.base_model.encode(
                documents, convert_to_tensor=True
            ).to(self.device)
            scores = F.cosine_similarity(
                query_embeddings.unsqueeze(1), doc_embeddings.unsqueeze(0), dim=2
            )

        return scores.cpu().numpy()


class OpenSearchTeacher:
    def __init__(self, local_rank):
        self.device = torch.device(f"cuda:{local_rank}")
        self.base_model = SentenceTransformer(
            "opensearch-project/opensearch-neural-sparse-encoding-v1"
        )
        self.base_model.to(self.device)
        self.model = DDP(self.base_model, device_ids=[local_rank])

    def __call__(self, queries, documents):
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(documents, str):
            documents = [documents]

        with torch.cuda.amp.autocast():
            query_embeddings = self.base_model.encode(
                queries, convert_to_tensor=True
            ).to(self.device)
            doc_embeddings = self.base_model.encode(
                documents, convert_to_tensor=True
            ).to(self.device)
            scores = torch.matmul(query_embeddings, doc_embeddings.t())

        return scores.cpu().numpy()


class MonoT5Teacher:
    def __init__(self, local_rank):

        self.device = torch.device(f"cuda:{local_rank}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "castorini/monot5-3b-msmarco-10k"
        )
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
            "castorini/monot5-3b-msmarco-10k"
        )
        self.base_model.to(self.device)
        self.model = DDP(self.base_model, device_ids=[local_rank])

    def __call__(self, queries, documents):
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(documents, str):
            documents = [documents]

        scores = []
        with torch.cuda.amp.autocast():
            for query, doc in zip(queries, documents):
                input_text = f"Query: {query} Document: {doc} Relevant:"
                inputs = self.tokenizer(
                    input_text, return_tensors="pt", truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.base_model.generate(
                        **inputs,
                        max_length=10,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    score = torch.softmax(outputs.scores[0][0], dim=0)[
                        self.tokenizer.encode("true")[0]
                    ]
                    scores.append(score.item())

        return torch.tensor(scores).numpy()


class CrossEncoderTeacher:
    def __init__(self, local_rank):
        self.device = torch.device(f"cuda:{local_rank}")
        self.base_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
        self.base_model.model = self.base_model.model.to(self.device)
        self.model = DDP(self.base_model.model, device_ids=[local_rank])

    def __call__(self, queries, documents):
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(documents, str):
            documents = [documents]

        pairs = [[query, doc] for query, doc in zip(queries, documents)]

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                scores = self.base_model.predict(pairs)

        if isinstance(scores, (float, int)):
            scores = np.array([scores])
        return np.array(scores)


def get_teacher(model_index, local_rank):
    if model_index == 0:
        return GTETeacher(local_rank)
    elif model_index == 1:
        return OpenSearchTeacher(local_rank)
    elif model_index == 2:
        return CrossEncoderTeacher(local_rank)
    elif model_index == 3:
        return MonoT5Teacher(local_rank)
    else:
        raise ValueError(f"Invalid model_index: {model_index}")


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
        run_res = search_result["run_res"]
        query_doc_scores = {}
        batch_size = 50

        for model_idx in range(4):
            if model_idx == 3:
                accelerator.set_floating_point_format("fp32")
            teacher = get_teacher(model_idx, local_rank)

            batch_queries = []
            batch_docs = []
            batch_query_ids = []
            batch_doc_ids = []

            for query, docs in tqdm(
                run_res.items(), desc=f"Processing with teacher {model_idx}"
            ):
                if query not in qrels:
                    continue

                # get the top 100 docs of each query
                doc_ids = [doc_id for doc_id, _ in docs[:100]]

                for doc_id in doc_ids:
                    batch_queries.append(queries[query])
                    batch_docs.append(corpus[doc_id]["text"])
                    batch_query_ids.append(query)

                    if len(batch_queries) == batch_size:
                        # Process batch
                        scores = process_batch_with_teacher(
                            teacher, batch_queries, batch_docs
                        )

                        # Store scores
                        for i in range(len(batch_queries)):
                            q_id = batch_query_ids[i]
                            d_id = batch_doc_ids[i]
                            if q_id not in query_doc_scores:
                                query_doc_scores[q_id] = {}
                            if d_id not in query_doc_scores[q_id]:
                                query_doc_scores[q_id][d_id] = []
                            query_doc_scores[q_id][d_id].append(scores[i])

                        batch_queries = []
                        batch_docs = []
                        batch_query_ids = []
                        batch_doc_ids = []

            # Process any remaining items (though unlikely with only 2 batches)
            if batch_queries:
                scores = process_batch_with_teacher(teacher, batch_queries, batch_docs)
                for i in range(len(batch_queries)):
                    q_id = batch_query_ids[i]
                    d_id = batch_doc_ids[i]
                    if q_id not in query_doc_scores:
                        query_doc_scores[q_id] = {}
                    if d_id not in query_doc_scores[q_id]:
                        query_doc_scores[q_id][d_id] = []
                    query_doc_scores[q_id][d_id].append(scores[i])

            del teacher
            torch.cuda.empty_cache()

        # Convert to final format
        final_data = []
        for query_id, doc_scores in query_doc_scores.items():
            query_entry = {"query": queries[query_id], "docs": [], "scores": []}

            for doc_id, scores in doc_scores.items():
                query_entry["docs"].append(corpus[doc_id]["text"])
                # Calculate average score, handling cases where not all models provided scores
                avg_score = np.mean(scores)
                query_entry["scores"].append(float(avg_score))

            final_data.append(query_entry)

        # Save as dataset
        ds = Dataset.from_list(final_data)
        ds.save_to_disk("data/msmarco_ft")


if __name__ == "__main__":
    local_rank, world_size = init_distributed()
    main()
