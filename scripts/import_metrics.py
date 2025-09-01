import argparse
import json
import os
import re
import sys
from datetime import datetime

from utils import emit_metrics


def infer_index_and_doc_id_from_path(file_path: str, root_dir: str):
    rel_path = os.path.relpath(file_path, root_dir)
    parts = rel_path.split(os.sep)
    # Try to infer from parent directory name and filename
    # Example:
    #   output/.../beir_eval_2p_0.1/avg_res.json -> index: beir_eval, doc_id: <dir_suffix>
    #   output/.../beir_eval_2p_0.1/beir_statictics.csv -> index: beir_eval_records
    #   output/.../nano_beir_eval_2p_0.1/avg_res_step125.json -> index: nano_beir_eval
    #   output/.../nano_beir_eval_2p_0.1/nano_beir_statictics_step125.csv -> index: nano_beir_eval_records

    parent = parts[-2] if len(parts) >= 2 else ""
    filename = parts[-1]

    # default doc_id: absolute dir + optional step suffix
    base_doc_id = os.path.abspath(os.path.dirname(file_path))

    # beir eval
    if parent.startswith("beir_eval"):
        if filename == "avg_res.json":
            return "beir_eval", base_doc_id
        if filename == "beir_statictics.csv":
            return "beir_eval_records", base_doc_id

    # nano beir eval
    if parent.startswith("nano_beir_eval"):
        if filename.startswith("avg_res_step") and filename.endswith(".json"):
            return "nano_beir_eval", base_doc_id
        if filename.startswith("nano_beir_statictics_step") and filename.endswith(
            ".csv"
        ):
            return "nano_beir_eval_records", base_doc_id

    return None, None


def find_experiment_output_dir(file_path: str) -> str:
    # Find the parent directory that contains the beir_eval or nano_beir_eval directory
    norm = os.path.normpath(file_path)
    parts = norm.split(os.sep)
    for i, part in enumerate(parts):
        if part.startswith("beir_eval") or part.startswith("nano_beir_eval"):
            # output_dir is the parent of this part
            return os.sep.join(parts[:i]) if i > 0 else os.sep
    # default to parent of the file
    return os.path.dirname(os.path.dirname(file_path))


def read_last_timestamp_from_log(output_dir: str):
    log_path = os.path.join(output_dir, "eval_beir.log")
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            # read from end: simple approach, read all and reverse iterate
            lines = f.read().splitlines()
        time_regex = re.compile(r"^(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})")
        for line in reversed(lines):
            m = time_regex.match(line.strip())
            if m:
                dt = datetime.strptime(m.group(1), "%m/%d/%Y %H:%M:%S")
                return dt.timestamp()
    except Exception:
        return None
    return None


def _extract_step_from_filename(filename: str):
    m = re.search(
        r"(?:avg_res_step|nano_beir_statictics_step)(\d+)\.(?:json|csv)$", filename
    )
    if m:
        return m.group(1)
    return None


def compute_dataset_number(index: str, file_path: str):
    try:
        import pandas as pd
    except Exception:
        return None

    dir_path = os.path.dirname(file_path)

    # beir eval
    if index == "beir_eval" or index == "beir_eval_records":
        csv_path = os.path.join(dir_path, "beir_statictics.csv")
        if os.path.isfile(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if "dataset" in df.columns:
                    return int(df["dataset"].nunique())
                return int(len(df))
            except Exception:
                return None
        return None

    # nano beir eval
    if index == "nano_beir_eval" or index == "nano_beir_eval_records":
        step = _extract_step_from_filename(os.path.basename(file_path))
        csv_path = None
        if step:
            csv_path = os.path.join(dir_path, f"nano_beir_statictics_step{step}.csv")
        else:
            # fallback: pick any matching csv if exists
            try:
                for name in os.listdir(dir_path):
                    if name.startswith("nano_beir_statictics_step") and name.endswith(
                        ".csv"
                    ):
                        csv_path = os.path.join(dir_path, name)
                        break
            except Exception:
                csv_path = None
        if csv_path and os.path.isfile(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if "dataset" in df.columns:
                    return int(df["dataset"].nunique())
                return int(len(df))
            except Exception:
                return None
        return None

    return None


def emit_file(file_path: str, root_dir: str):
    index, doc_id = infer_index_and_doc_id_from_path(file_path, root_dir)
    if index is None:
        return 0

    # Prefer timestamp from the corresponding eval_beir.log in output_dir
    output_dir = find_experiment_output_dir(file_path)
    timestamp = read_last_timestamp_from_log(output_dir)
    if timestamp is None:
        timestamp = datetime.now().timestamp()

    dataset_number = compute_dataset_number(index, file_path)

    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            metrics = json.load(f)
        # ensure timestamp field
        if "timestamp" not in metrics:
            metrics["timestamp"] = timestamp
        else:
            # override with log timestamp if available
            if timestamp is not None:
                metrics["timestamp"] = timestamp
        # only add dataset_number for aggregate indices
        if index == "beir_eval" or index == "nano_beir_eval":
            if dataset_number is not None:
                metrics["dataset_number"] = dataset_number
        emit_metrics(metrics, index, doc_id)
        return 1

    if file_path.endswith(".csv"):
        # load csv and emit as records
        try:
            import pandas as pd
        except Exception:
            print("pandas is required to import CSV metrics")
            return 0
        df = pd.read_csv(file_path)
        records = df.to_dict(orient="records")
        metrics = {"records": records, "timestamp": timestamp}
        # keep parity with evaluate_beir: records emit does not include dataset_number
        emit_metrics(metrics, index, doc_id)
        return 1

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Import existing metrics into OpenSearch indices recursively"
    )
    parser.add_argument("dir", help="Root directory to scan recursively")
    args = parser.parse_args()

    root = args.dir
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        sys.exit(1)

    imported = 0
    for current_dir, _, files in os.walk(root):
        for name in files:
            if not (name.endswith(".json") or name.endswith(".csv")):
                continue
            imported += emit_file(os.path.join(current_dir, name), root)

    print(f"Imported {imported} metric files into indices")


if __name__ == "__main__":
    main()
