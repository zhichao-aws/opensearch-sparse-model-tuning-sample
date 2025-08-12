import os
import threading
import numpy as np
import logging

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import List, Dict, Any, Optional
from .dynamo_utils import batch_get_vectors

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, max_workers=10):
        """Initialize the embedding service with given number of worker threads"""
        # Check for required environment variables
        if "AWS_REGION_NAME" not in os.environ:
            raise EnvironmentError("AWS_REGION_NAME environment variable is required")

        self.region_name = os.environ.get("AWS_REGION_NAME")

        # Initialize internal state
        self.registered_tasks = {}
        self.fetched_embeddings = {}
        self.task_events = {}
        self.lock = threading.Lock()
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)

    def fetch_embeddings_from_dynamodb(
        self, table_name: str, model_id: int, ids: List[int]
    ):
        """Background task to fetch embeddings from DynamoDB"""
        try:
            embeddings = batch_get_vectors(
                table_name, ids, model_id=model_id, region_name=self.region_name
            )
            # Store results and set event flag
            with self.lock:
                key = f"{table_name}_{model_id}_{','.join(map(str, ids))}"
                self.fetched_embeddings[key] = embeddings
                if key in self.task_events:  # Ensure task still exists
                    self.task_events[key].set()
        except Exception as e:
            # Log error and set event to avoid deadlocks
            with self.lock:
                key = f"{table_name}_{model_id}_{','.join(map(str, ids))}"
                self.fetched_embeddings[key] = {"error": str(e)}
                if key in self.task_events:
                    self.task_events[key].set()

    def register_task(
        self, table_name: str, model_id: int, ids: List[int]
    ) -> Dict[str, Any]:
        """Register a task to fetch embeddings"""
        key = f"{table_name}_{model_id}_{','.join(map(str, ids))}"

        with self.lock:
            # Update or create task
            self.registered_tasks[key] = self.registered_tasks.get(key, 0) + 1

            # If it's a new task, create event and submit to thread pool
            if key not in self.task_events:
                self.task_events[key] = Event()
                needs_submit = True
            else:
                needs_submit = False

        # Submit task outside the lock
        if needs_submit:
            self.thread_pool.submit(
                self.fetch_embeddings_from_dynamodb,
                table_name,
                model_id,
                ids,
            )

        return {"status": "success", "task_id": key}

    def fetch_embedding(
        self, table_name: str, model_id: int, ids: List[int]
    ) -> np.ndarray:
        """Fetch embeddings, wait if task not completed"""
        key = f"{table_name}_{model_id}_{','.join(map(str, ids))}"

        # Check if task exists
        with self.lock:
            if key not in self.registered_tasks:
                raise ValueError("Task not registered")

            # If results are available, return directly
            if key in self.fetched_embeddings:
                embeddings = self.fetched_embeddings[key]
                event = None
            else:
                # Get event object to wait outside the lock
                event = self.task_events[key]
                embeddings = None

        # Wait for task completion outside the lock
        if event:
            event.wait()
            # Get results after waiting
            with self.lock:
                if key in self.fetched_embeddings:
                    embeddings = self.fetched_embeddings[key]
                else:
                    raise RuntimeError("Task result not available")

        # Decrease task count and clean up completed tasks
        with self.lock:
            self.registered_tasks[key] -= 1
            if self.registered_tasks[key] <= 0:
                # Clean up completed tasks
                self.registered_tasks.pop(key, None)
                self.fetched_embeddings.pop(key, None)
                self.task_events.pop(key, None)

        # Check if result contains error information
        if isinstance(embeddings, dict) and "error" in embeddings:
            raise RuntimeError(f"Task failed: {embeddings['error']}")

        return embeddings

    def health_check(self) -> Dict[str, str]:
        """Health check method"""
        return {"status": "healthy"}

    def shutdown(self):
        """Clean up resources when shutting down"""
        self.thread_pool.shutdown(wait=True)
