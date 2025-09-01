import logging
import os

import torch
from tqdm import tqdm

from .dataset.dataset import KeyValueDataset
from .model.sparse_encoders import SparseEncoder, SparseModel, sparse_embedding_to_query
from .utils import batch_search, get_os_client

logger = logging.getLogger(__name__)


async def search(
    queries: dict,
    model: SparseModel,
    out_dir: str,
    index_name: str,
    max_length: int = 512,
    batch_size: int = 50,
    result_size: int = 15,
    inf_free: bool = True,
    delete: bool = False,
    use_two_phase: bool = False,
    query_prune: float = 0,
    return_text: bool = False,
):
    if use_two_phase:
        client = get_os_client()
        client.transport.perform_request(
            "PUT",
            "/_search/pipeline/neural_search_pipeline",
            body={
                "request_processors": [
                    {
                        "neural_sparse_two_phase_processor": {
                            "tag": "neural-sparse",
                            "description": "This processor creates a neural sparse two-phase processor, which can speed up neural sparse queries!",
                        }
                    }
                ]
            },
        )

    os.makedirs(out_dir, exist_ok=True)

    queries_dataset = KeyValueDataset(queries)
    dataloader = torch.utils.data.DataLoader(queries_dataset, batch_size=batch_size)

    query_encoder = SparseEncoder(
        sparse_model=model,
        max_length=max_length,
        do_count=True,
    )

    run_res = dict()
    for ids, texts in tqdm(dataloader):
        queries_encoded = query_encoder.encode(texts, inf_free=inf_free)

        search_results = await batch_search(
            queries=queries_encoded,
            index_name=index_name,
            endpoint_lambda=lambda index_name: f"""http://localhost:9200/{index_name}/_search""",
            get_query_lambda=lambda query: {
                "size": result_size,
                "query": sparse_embedding_to_query(query, query_prune=query_prune),
                "_source": ["id", "text"],
            },
            interval=0.001,
            use_two_phase=use_two_phase,
        )

        for i, (_id, res) in enumerate(zip(ids, search_results)):
            if return_text:
                run_res[_id] = [hit["_source"]["text"] for hit in res]
            else:
                run_res[_id] = {hit["_source"]["id"]: hit["_score"] for hit in res}

    for query_id, doc_dict in run_res.items():
        if query_id in doc_dict:
            doc_dict.pop(query_id)

    count = query_encoder.count_tensor
    count = count / len(queries_dataset)
    count_doc = torch.load(
        os.path.join(out_dir, index_name + ".corpus.bin"),
        map_location=model.backbone.device,
    )
    flops = float(torch.matmul(count, count_doc).cpu())
    q_length = float(count.sum().cpu())
    d_length = float(count_doc.sum().cpu())
    logger.info(
        f"Index_name: {index_name}, flops: {flops}, d_length:{d_length}, q_length:{q_length}"
    )

    if delete:
        client = get_os_client()
        client.indices.delete(index_name, params={"timeout": 1000})

    return {
        "run_res": run_res,
        "flops": flops,
        "q_length": q_length,
        "d_length": d_length,
    }
