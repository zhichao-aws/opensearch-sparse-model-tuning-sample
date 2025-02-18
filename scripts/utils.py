import os
import asyncio
import aiohttp
from aiohttp import ClientTimeout
import json
import logging
import sys
import torch
import torch.distributed as dist

from opensearchpy import OpenSearch
from .model.sparse_encoders import SparseModel


def gather_rep(rep, accelerator):
    if accelerator.num_processes == 1:
        return rep
    all_rep = accelerator.gather(rep)
    size = rep.shape[0]
    local_index = accelerator.local_process_index
    all_rep[local_index * size : local_index * size + size] = rep
    return all_rep


def is_ddp_enabled():
    if dist.is_initialized():
        return True
    return False


def get_os_client():
    return OpenSearch(
        hosts=["http://localhost:9200"], verify_certs=False, ssl_show_warn=False
    )


def set_logging(training_args, log_file_name):
    logging.basicConfig(
        level=training_args.get_process_log_level(),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(training_args.output_dir, log_file_name)),
        ],
    )


def get_model(model_args):
    idf = None
    if model_args.inf_free and model_args.idf_path:
        with open(model_args.idf_path) as f:
            idf = json.load(f)

    model = SparseModel(
        model_args.model_name_or_path,
        idf=idf,
        tokenizer_id=model_args.tokenizer_name,
        split_batch=model_args.split_batch,
        idf_requires_grad=model_args.idf_requires_grad,
        activation_type=model_args.activation_type,
        model_type=model_args.model_type,
    )

    return model


async def do_search(session, endpoint, query_body, post_process=None):
    url = endpoint
    body = query_body

    async with session.get(url, json=body, verify_ssl=False) as resp:
        response = await resp.json()
        if "error" in response:
            raise Exception(response["error"])

    hits = response["hits"]["hits"]
    if post_process is not None:
        hits = post_process(hits)
    return hits


async def do_bulk(bulk_body, session, endpoint="http://localhost:9200"):
    url = endpoint + "/_bulk"
    bulk_body = "\n".join(map(json.dumps, bulk_body)) + "\n"
    headers = {"Content-Type": "application/x-ndjson"}
    async with session.post(url, data=bulk_body, headers=headers) as resp:
        response = await resp.json()
        if "errors" not in response:
            print(response)
        try:
            assert response["errors"] == False
        except:
            print(response)
            raise Exception("bulk request failed")

    return response


async def batch_search(
    queries,
    index_name,
    endpoint_lambda,
    get_query_lambda,
    post_process=None,
    interval=0.01,
):
    timeout = ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for query in queries:
            tasks.append(
                asyncio.create_task(
                    do_search(
                        session,
                        endpoint_lambda(index_name),
                        get_query_lambda(query),
                        post_process=post_process,
                    )
                )
            )
            await asyncio.sleep(interval)

        try:
            result = await asyncio.gather(*tasks)
        except Exception as e:
            print("error happens when querying index ", index_name)
            print(e)
            return {"error": str(e)}
    return result


def get_logger_console(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
