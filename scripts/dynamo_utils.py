import time

import boto3
import numpy as np


def batch_store_vectors_binary(
    table_name,
    vector_ids,
    vectors,
    model_id=0,
    max_batch_size=25,
    region_name="ap-northeast-1",
):
    if len(vector_ids) != vectors.shape[0]:
        raise ValueError(
            f"Vector IDs {len(vector_ids)} not equal to vectors {vectors.shape[0]}"
        )

    total_vectors = len(vector_ids)
    dynamodb_client = boto3.client("dynamodb", region_name=region_name)

    for batch_start in range(0, total_vectors, max_batch_size):
        batch_end = min(batch_start + max_batch_size, total_vectors)
        batch_items = []

        for i in range(batch_start, batch_end):
            vector = vectors[i]
            vector_bytes = vector.tobytes()

            item = {
                "PutRequest": {
                    "Item": {
                        "text_id": {"N": str(vector_ids[i])},
                        "model_id": {"N": str(model_id)},
                        "vector": {"B": vector_bytes},
                    }
                }
            }
            batch_items.append(item)

        if batch_items:
            response = dynamodb_client.batch_write_item(
                RequestItems={table_name: batch_items}
            )

            unprocessed_items = response.get("UnprocessedItems", {})
            retry_attempts = 0
            max_retries = 100

            while unprocessed_items and retry_attempts < max_retries:
                time.sleep(0.5)
                retry_attempts += 1
                print(f"Attempt {retry_attempts}...")
                response = dynamodb_client.batch_write_item(
                    RequestItems=unprocessed_items
                )
                unprocessed_items = response.get("UnprocessedItems", {})

            if unprocessed_items:
                raise Exception(
                    f"Warning: {len(unprocessed_items.get(table_name, []))} items failed to write"
                )


def batch_get_vectors(
    table_name,
    text_id_list,
    model_id=0,
    dtype=np.float16,
    max_batch_size=100,
    region_name="ap-northeast-1",
):
    if not text_id_list:
        return []

    text_id_list = [int(x) for x in text_id_list]
    dynamodb_client = boto3.client("dynamodb", region_name=region_name)
    model_id_str = str(model_id)
    result_vectors = [None] * len(
        text_id_list
    )  # Pre-allocate result list to maintain input order

    # Deduplicate text_id_list while preserving original index information
    unique_text_ids = []
    original_to_unique_map = {}  # Mapping from original list index to deduplicated list index

    for idx, text_id in enumerate(text_id_list):
        if text_id not in original_to_unique_map:
            original_to_unique_map[text_id] = len(unique_text_ids)
            unique_text_ids.append(text_id)

    # Create a dictionary to store retrieved vectors, using text_id as key
    id_to_vector = {}

    # Process deduplicated ID list in batches
    for batch_start in range(0, len(unique_text_ids), max_batch_size):
        batch_end = min(batch_start + max_batch_size, len(unique_text_ids))
        batch_ids = unique_text_ids[batch_start:batch_end]

        # Build batch get request
        keys_list = [
            {"text_id": {"N": str(text_id)}, "model_id": {"N": model_id_str}}
            for text_id in batch_ids
        ]

        response = dynamodb_client.batch_get_item(
            RequestItems={
                table_name: {
                    "Keys": keys_list,
                    "ConsistentRead": True,  # Ensure read consistency
                }
            }
        )

        # Process response
        if "Responses" in response and table_name in response["Responses"]:
            items = response["Responses"][table_name]

            # Process each returned item
            for item in items:
                text_id = int(item["text_id"]["N"])
                vector_binary = item["vector"]["B"]
                vector = np.frombuffer(vector_binary, dtype=dtype)

                # Store vector using text_id as key
                id_to_vector[text_id] = vector

        # Handle unprocessed keys
        unprocessed_keys = (
            response.get("UnprocessedKeys", {}).get(table_name, {}).get("Keys", [])
        )
        retry_attempts = 0
        max_retries = 10

        # Retry getting unprocessed keys
        while unprocessed_keys and retry_attempts < max_retries:
            time.sleep(0.5)

            retry_attempts += 1
            print(f"Retrying unprocessed keys, attempt {retry_attempts}...")

            # Only request unprocessed keys
            retry_response = dynamodb_client.batch_get_item(
                RequestItems={
                    table_name: {"Keys": unprocessed_keys, "ConsistentRead": True}
                }
            )

            if (
                "Responses" in retry_response
                and table_name in retry_response["Responses"]
            ):
                retry_items = retry_response["Responses"][table_name]

                for item in retry_items:
                    text_id = int(item["text_id"]["N"])
                    vector_binary = item["vector"]["B"]
                    vector = np.frombuffer(vector_binary, dtype=dtype)

                    # Store vector
                    id_to_vector[text_id] = vector

            unprocessed_keys = (
                retry_response.get("UnprocessedKeys", {})
                .get(table_name, {})
                .get("Keys", [])
            )

        if unprocessed_keys:
            raise Exception(f"Warning: {len(unprocessed_keys)} keys failed to read")

    # Fill result_vectors according to the original text_id_list order
    for idx, text_id in enumerate(text_id_list):
        if text_id in id_to_vector:
            result_vectors[idx] = id_to_vector[text_id]

    return np.array(result_vectors)
