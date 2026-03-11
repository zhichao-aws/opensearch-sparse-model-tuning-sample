import asyncio
import logging

import aiohttp
from accelerate import Accelerator
from aiohttp import ClientTimeout
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .dataset.dataset import DDPDatasetWithRank
from .utils import do_bulk, get_os_client

logger = logging.getLogger(__name__)


async def ingest_dense(
    dataset: Dataset,
    model_id: str,
    index_name: str,
    accelerator: Accelerator,
    batch_size: int = 64,
    max_length: int = 512,
):
    os_client = get_os_client()

    # Check dataset type
    if isinstance(dataset, DDPDatasetWithRank):
        logger.error("Input dataset can not be DDPDatasetWithRank.")
        raise RuntimeError("Input dataset can not be DDPDatasetWithRank.")

    # Wrap dataset for DDP
    ddp_dataset = DDPDatasetWithRank(
        dataset, accelerator.local_process_index, accelerator.num_processes
    )
    dataloader = DataLoader(
        ddp_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: list(zip(*x)),
    )

    logger.info(
        f"Local rank: {accelerator.local_process_index}, index_name: {index_name}, sample number: {len(ddp_dataset)}"
    )

    # Load model on correct device
    logger.info(
        f"Loading SentenceTransformer model: {model_id} on {accelerator.device}"
    )
    model = SentenceTransformer(model_id, device=accelerator.device, trust_remote_code=True)
    model.max_seq_length = max_length

    # Get dimension
    embedding_dim = model.get_sentence_embedding_dimension()
    if accelerator.is_local_main_process:
        logger.info(f"Model dimension: {embedding_dim}")

    # Prepare index (only on main process)
    if accelerator.is_local_main_process:
        try:
            if os_client.indices.exists(index=index_name):
                os_client.indices.delete(index=index_name)
            # delete the index if exist - following ingest.py pattern or check existence
            if not os_client.indices.exists(index=index_name):
                logger.info(f"Creating index {index_name} with HNSW")
                body = {
                    "settings": {
                        "index": {
                            "number_of_shards": 12,  # Increased shard count for better distribution if needed, matching ingest.py default often
                            "number_of_replicas": 0,
                        }
                    },
                    "mappings": {
                        "properties": {
                            "embedding": {
                                "type": "knn_vector",
                                "dimension": embedding_dim
                            },
                            "text": {"type": "text"},
                            "id": {"type": "keyword"},
                        }
                    },
                }
                os_client.indices.create(index=index_name, body=body)
        except Exception as e:
            logger.warning(f"Index creation failed (might exist): {e}")

    accelerator.wait_for_everyone()

    # Ingest
    logger.info("Starting ingestion...")

    tasks = []
    timeout = ClientTimeout(total=600)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            ids = batch[0]
            texts = batch[1]

            # Encode
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            bulk_body = []
            for i in range(len(ids)):
                bulk_body.append({"index": {"_index": index_name, "_id": ids[i]}})
                bulk_body.append(
                    {
                        "text": texts[i],
                        "embedding": embeddings[i].tolist(),
                        "id": ids[i],
                    }
                )

            # Async bulk
            tasks.append(asyncio.create_task(do_bulk(bulk_body, session)))

            # Rate limit / concurrent limit
            if len(tasks) >= 20:
                await asyncio.gather(*tasks)
                tasks = []

        if tasks:
            await asyncio.gather(*tasks)

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        logger.info("Ingestion complete. Refreshing index...")
        os_client.indices.refresh(index=index_name, params={"timeout": 1000})
