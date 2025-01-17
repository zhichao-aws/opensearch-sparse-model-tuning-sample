import os
import torch
import torch.distributed as dist
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import time
from tqdm import tqdm
from data_preparation import get_teacher


def setup_test_data():
    """Create test data with known relationships"""
    test_cases = [
        # Highly relevant pairs
        (
            "What is machine learning?",
            "Machine learning is a branch of artificial intelligence that enables computers to learn from data.",
            1,
        ),
        (
            "Who was Albert Einstein?",
            "Albert Einstein was a renowned physicist who developed the theory of relativity.",
            1,
        ),
        # Somewhat relevant pairs
        (
            "What are the benefits of exercise?",
            "Regular physical activity can improve health and mood.",
            0.7,
        ),
        # Irrelevant pairs
        (
            "What is the capital of France?",
            "Bananas are rich in potassium and other nutrients.",
            0,
        ),
        (
            "How does photosynthesis work?",
            "The Great Wall of China is over 13,000 miles long.",
            0,
        ),
    ]

    queries, documents, relevance_scores = zip(*test_cases)
    return list(queries), list(documents), list(relevance_scores)


def evaluate_predictions(predictions, ground_truth, threshold=0.5):
    """Evaluate model predictions"""
    # Convert scores to binary predictions using threshold
    binary_preds = (predictions >= threshold).astype(int)
    binary_truth = (np.array(ground_truth) >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(binary_truth, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_truth, binary_preds, average="binary"
    )

    # Calculate correlation
    correlation = np.corrcoef(predictions, ground_truth)[0, 1]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correlation": correlation,
    }


def run_tests(teacher, local_rank):
    """Run comprehensive tests on a teacher model"""
    logging.info(f"Starting tests on GPU {local_rank}")

    # Get test data
    queries, documents, ground_truth = setup_test_data()

    results = {}

    # Test 1: Basic Functionality
    try:
        start_time = time.time()
        scores = teacher(queries, documents)
        inference_time = time.time() - start_time

        # Normalize scores to 0-1 range if needed
        scores = (scores - scores.min()) / (scores.max() - scores.min())

        metrics = evaluate_predictions(scores, ground_truth)

        results["basic_test"] = {
            "success": True,
            "metrics": metrics,
            "inference_time": inference_time,
            "scores": scores.tolist(),
        }

    except Exception as e:
        results["basic_test"] = {"success": False, "error": str(e)}

    # Test 2: Batch Size Stress Test
    batch_sizes = [1, 2, 4, 8, 16]
    batch_results = {}

    for batch_size in batch_sizes:
        try:
            # Create larger batch by repeating data
            batch_queries = queries * batch_size
            batch_documents = documents * batch_size

            start_time = time.time()
            batch_scores = teacher(batch_queries, batch_documents)
            batch_time = time.time() - start_time

            batch_results[batch_size] = {
                "success": True,
                "time_per_item": batch_time / len(batch_queries),
                "total_time": batch_time,
            }

        except Exception as e:
            batch_results[batch_size] = {"success": False, "error": str(e)}

    results["batch_test"] = batch_results

    # Test 3: Edge Cases
    edge_cases = {
        "empty_query": ("", "This is a document", None),
        "empty_doc": ("What is this?", "", None),
        "long_query": ("what " * 1000, "document", None),
        "long_doc": ("query", "word " * 1000, None),
        "special_chars": (
            "What is this?!@#$%^&*()",
            "Document with special chars ?!@#$%^&*()",
            None,
        ),
    }

    edge_results = {}
    for case_name, (query, doc, _) in edge_cases.items():
        try:
            score = teacher([query], [doc])
            edge_results[case_name] = {"success": True, "score": score.tolist()}
        except Exception as e:
            edge_results[case_name] = {"success": False, "error": str(e)}

    results["edge_cases"] = edge_results

    return results


def main():
    # Initialize distributed training
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - GPU#{} - %(message)s".format(local_rank),
        level=logging.INFO if local_rank in [-1, 0] else logging.WARNING,
    )

    try:
        # Test all teacher models
        for model_index in range(
            4
        ):  # 0: GTE, 1: OpenSearch, 2: MonoT5, 3: CrossEncoder
            if model_index != 2:
                continue
            teacher = get_teacher(model_index, local_rank)
            results = run_tests(teacher, local_rank)

            if local_rank == 0:  # Only print results from main process
                print(f"\n=== Results for Model {model_index} ===")
                print("\nBasic Test Results:")
                if results["basic_test"]["success"]:
                    metrics = results["basic_test"]["metrics"]
                    print(f"Accuracy: {metrics['accuracy']:.3f}")
                    print(f"F1 Score: {metrics['f1']:.3f}")
                    print(f"Correlation: {metrics['correlation']:.3f}")
                    print(
                        f"Inference time: {results['basic_test']['inference_time']:.3f}s"
                    )
                else:
                    print(f"Error: {results['basic_test']['error']}")

                print("\nBatch Size Test Results:")
                for batch_size, batch_result in results["batch_test"].items():
                    if batch_result["success"]:
                        print(
                            f"Batch size {batch_size}: {batch_result['time_per_item']:.4f}s per item"
                        )
                    else:
                        print(
                            f"Batch size {batch_size}: Failed - {batch_result['error']}"
                        )

                print("\nEdge Case Results:")
                for case, result in results["edge_cases"].items():
                    status = "✓" if result["success"] else "✗"
                    print(f"{case}: {status}")

            # after each model done, remove it from GPU to free up memory
            del teacher
            torch.cuda.empty_cache()

    except Exception as e:
        logging.error(f"Error on GPU {local_rank}: {str(e)}")
        raise e

    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
