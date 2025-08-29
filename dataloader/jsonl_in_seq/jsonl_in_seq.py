# Local JSONL dataset loader for Hugging Face Datasets
# This loader supports:
# - Single or multiple JSONL files (provided via data/manifest.txt)
# - Streaming mode (IterableDataset)
# - On-the-fly split: validation takes 1% (every 100th line), train takes the rest

import gzip
import json
import os
from typing import List, Iterator, Tuple

import datasets


logger = datasets.logging.get_logger(__name__)


class JsonlLocalConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class JsonlLocal(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        JsonlLocalConfig(name="default", description="Local JSONL dataset with 1% validation (every 100th line)"),
    ]

    DEFAULT_CONFIG_NAME = "default"

    def _info(self) -> datasets.DatasetInfo:
        return datasets.DatasetInfo(
            description="Local JSONL dataset (text-only), validation is every 100th line",
            features=datasets.Features({
                "text": datasets.Value("string"),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        manifest_path = "manifest.txt"

        # Allow overriding manifest via env, or directly pass file list via env.
        env_manifest = os.environ.get("JSONL_LOCAL_MANIFEST")
        env_files = os.environ.get("JSONL_LOCAL_FILES")

        files = None
        if env_files:
            # Support comma or colon separated absolute paths
            sep = "," if "," in env_files else ":"
            candidates = [p.strip() for p in env_files.split(sep) if p.strip()]
            files = candidates
        else:
            if env_manifest and os.path.exists(env_manifest):
                manifest_path = env_manifest
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(
                    f"manifest.txt not found at: {manifest_path}. Provide it or set env JSONL_LOCAL_MANIFEST, "
                    f"or set JSONL_LOCAL_FILES with absolute file paths (comma/colon separated)."
                )
            with open(manifest_path, "rt", encoding="utf-8") as f:
                files = [line.strip() for line in f if line.strip()]

        # No download: files are local paths listed in manifest
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": files, "split_name": "train"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"files": files, "split_name": "validation"}),
        ]

    def _generate_examples(self, files: List[str], split_name: str) -> Iterator[Tuple[str, dict]]:
        # validation: every 100th line (global index % 100 == 0)
        # train: the rest
        global_index = 0
        for file_idx, fp in enumerate(files):
            if not os.path.exists(fp):
                logger.warning(f"File not found, skipping: {fp}")
                continue

            open_fn = gzip.open if fp.endswith(".gz") else open
            mode = "rt"
            with open_fn(fp, mode=mode, encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        global_index += 1
                        continue
                    try:
                        row = json.loads(line)
                    except Exception as e:
                        logger.warning(f"JSON decode error at {fp}:{line_idx}: {e}")
                        global_index += 1
                        continue

                    if "text" not in row:
                        global_index += 1
                        continue

                    is_validation = (global_index % 100 == 0)
                    if split_name == "validation" and not is_validation:
                        global_index += 1
                        continue
                    if split_name == "train" and is_validation:
                        global_index += 1
                        continue

                    key = f"{file_idx}-{line_idx}"
                    yield key, {"text": row["text"]}
                    global_index += 1
