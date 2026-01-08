import argparse
import os

import mteb
from sentence_transformers import SparseEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model", type=str, default=None, help="Base model name (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the training output directory",
    )
    parser.add_argument(
        "--max_active_dims",
        type=int,
        default=None,
        help="Keep top-K active dims in SPLADE output",
    )

    args = parser.parse_args()

    # Determine checkpoint path based on output_dir
    # Assume training script saved to output_dir/final
    model_path = os.path.join(args.output_dir, "final")
    print(f"Loading model from: {model_path}")

    model = SparseEncoder(model_path)
    tasks = mteb.get_tasks(tasks=["ChemHotpotQARetrieval", "ChemNQRetrieval"])
    evaluation = mteb.MTEB(tasks=tasks)

    # Determine evaluation result storage path based on output_dir
    # Maintain structure similar to original: ./results_mteb_chem/{model_name}
    # Simple approach is to use the name of output_dir directly
    model_name = os.path.basename(args.output_dir.rstrip("/"))
    eval_output_folder = f"./results_mteb_chem/{model_name}{args.max_active_dims if args.max_active_dims else ''}"

    # Note: MTEB does not currently support sparse tensors, so set convert_to_sparse_tensor=False
    evaluation.run(
        model,
        eval_splits=["test"],
        output_folder=eval_output_folder,
        show_progress_bar=True,
        encode_kwargs={
            "batch_size": 16,
            "convert_to_sparse_tensor": False,
            "max_active_dims": args.max_active_dims,
        },
    )


if __name__ == "__main__":
    main()
