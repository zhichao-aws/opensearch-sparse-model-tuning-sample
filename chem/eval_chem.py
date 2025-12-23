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

    # 根据 output_dir 确定 checkpoint 路径
    # 假设 training script 保存到了 output_dir/final
    model_path = os.path.join(args.output_dir, "final")
    print(f"Loading model from: {model_path}")

    model = SparseEncoder(model_path)
    tasks = mteb.get_tasks(tasks=["ChemHotpotQARetrieval", "ChemNQRetrieval"])
    evaluation = mteb.MTEB(tasks=tasks)

    # 根据 output_dir 确定评估结果保存路径
    # 保持类似原有的结构： ./results_mteb_chem/{model_name}
    # 简单的做法是直接用 output_dir 的名字
    model_name = os.path.basename(args.output_dir.rstrip("/"))
    eval_output_folder = f"./results_mteb_chem/{model_name}{args.max_active_dims if args.max_active_dims else ''}"

    # 注意：MTEB 目前不支持 sparse tensor，所以要 convert_to_sparse_tensor=False :contentReference[oaicite:2]{index=2}
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
