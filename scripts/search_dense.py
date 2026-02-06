import logging
import os

import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset.dataset import KeyValueDataset
from .utils import batch_search

logger = logging.getLogger(__name__)


async def search_dense(
    queries: dict,
    model_id: str,
    index_name: str,
    out_dir: str,
    device: torch.device,
    batch_size: int = 32,
    result_size: int = 10,
    return_text: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    logger.info(f"Loading SentenceTransformer model for search: {model_id} on {device}")
    model = SentenceTransformer(model_id, device=device)

    queries_dataset = KeyValueDataset(queries)
    dataloader = DataLoader(queries_dataset, batch_size=batch_size, shuffle=False)

    run_res = dict()

    logger.info("Starting dense search...")

    for ids, texts in tqdm(dataloader):
        # Encode queries
        embeddings = model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )

        search_results = await batch_search(
            queries=embeddings,
            index_name=index_name,
            endpoint_lambda=lambda index_name: (
                f"http://localhost:9200/{index_name}/_search"
            ),
            get_query_lambda=lambda query_vec: {
                "size": result_size,
                "query": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "knn_score",
                            "lang": "knn",
                            "params": {
                                "field": "embedding",
                                "query_value": query_vec.tolist(),
                                "space_type": "cosinesimil"
                            }
                        }
                    }
                },
                "_source": ["id"],
            },
            interval=0.001,
        )

        # Check if batch_search returned an error
        if isinstance(search_results, dict) and "error" in search_results:
            logger.error(f"Search failed: {search_results['error']}")
            raise Exception(f"Search failed: {search_results['error']}")

        for i, (_id, res) in enumerate(zip(ids, search_results)):
            if return_text:
                run_res[_id] = [hit["_source"]["text"] for hit in res]
            else:
                run_res[_id] = {hit["_source"]["id"]: hit["_score"] for hit in res}

    return {"run_res": run_res}
