import os
import os.path

import asyncio
import torch
import logging

from tqdm.auto import tqdm

from .model.sparse_encoders import SparseModel, SparseEncoder
from .utils import do_bulk, get_os_client
from .data.dataset import DDPDatasetWithRank
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator

import aiohttp
from aiohttp import ClientTimeout

logger = logging.getLogger(__name__)


async def ingest(
    dataset: Dataset,
    model: SparseModel,
    out_dir: str,
    index_name: str,
    accelerator: Accelerator,
    max_length: int = 512,
    batch_size: int = 50,
):
    os_client = get_os_client()
    os.makedirs(out_dir, exist_ok=True)
    # prepare DDP dataset and model
    # if not is_ddp_enabled():
    #     logger.error("The ingest can only work on environment that DDP is enabled.")
    #     raise RuntimeError(
    #         "The ingest can only work on environment that DDP is enabled."
    #     )
    if isinstance(dataset, DDPDatasetWithRank):
        logger.error("Input dataset can not be DDPDatasetWithRank.")
        raise RuntimeError("Input dataset can not be DDPDatasetWithRank.")
    ddp_dataset = DDPDatasetWithRank(
        dataset, accelerator.local_process_index, accelerator.num_processes
    )
    logger.info(
        f"Local rank: {accelerator.local_process_index}, index_name: {index_name}, sample number: {len(ddp_dataset)}"
    )
    dataloader = DataLoader(ddp_dataset, batch_size=batch_size)

    accelerator.prepare(model)
    sparse_encoder = SparseEncoder(
        sparse_model=model,
        max_length=max_length,
        do_count=True,
    )

    # prepare index
    if accelerator.is_local_main_process:
        try:
            # delete the index if exist
            os_client.indices.delete(index_name)
        except:
            pass

        os_client.indices.create(
            index=index_name,
            body={
                "settings": {
                    "index": {"number_of_shards": 12, "number_of_replicas": 0}
                },
                "mappings": {
                    # "_source": {"excludes": ["text_sparse"]},
                    "properties": {
                        "text_sparse": {"type": "rank_features"},
                        "text": {"type": "text"},
                        "id": {"type": "keyword"},
                    },
                },
            },
            params={"timeout": 1000},
        )
    accelerator.wait_for_everyone()

    # do model encoding and ingestion
    # use async io so we don't need to wait every bulk return
    # we send out 20 bulk request, then wait all of them return
    tasks = []
    timeout = ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for ids, texts in tqdm(dataloader):
            output = sparse_encoder.encode(texts)
            bulk_body = []
            for i in range(len(ids)):
                bulk_body.append({"index": {"_index": index_name, "_id": ids[i]}})
                bulk_body.append(
                    {"text": texts[i], "text_sparse": output[i], "id": ids[i]}
                )

            tasks.append(asyncio.create_task(do_bulk(bulk_body, session)))
            await asyncio.sleep(0.0001)
            if len(tasks) == 20:
                await asyncio.gather(*tasks)
                tasks = []

        await asyncio.gather(*tasks)

    sparse_encoder.count_tensor = sparse_encoder.count_tensor.reshape(1, -1)
    accelerator.wait_for_everyone()
    all_count_tensor = accelerator.gather(sparse_encoder.count_tensor)

    if accelerator.is_local_main_process:
        os_client.indices.refresh(index_name, params={"timeout": 1000})
        all_count_tensor = all_count_tensor.sum(dim=0)
        all_count_tensor = all_count_tensor / len(dataset)
        all_count_tensor = all_count_tensor.cpu()
        torch.save(all_count_tensor, os.path.join(out_dir, index_name + ".corpus.bin"))
