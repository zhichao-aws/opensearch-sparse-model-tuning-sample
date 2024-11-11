import os
import json
import logging
import random

from ..utils import is_ddp_enabled

from itertools import chain
from tqdm import tqdm
from datasets import load_dataset, Dataset as DatasetsDataset
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader, Sampler, RandomSampler, BatchSampler

logger = logging.getLogger(__name__)


class KeyValueDataset(Dataset):
    def __init__(self, data_dict):
        """
        Args:
            data_dict (dict): The input data dictionary where the keys are IDs and values are the content.
        """
        self.keys = sorted(data_dict.keys())
        self.data = {key: data_dict[key] for key in self.keys}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        value = self.data[key]
        return key, value


class BEIRCorpusDataset(KeyValueDataset):
    def __init__(self, corpus):
        """
        Args:
            corpus (dict): The corpus dictionary where each key is a document ID, and the value is a dictionary with "title" and "text".
        """
        # Filter out corpus entries where both title and text are empty
        filtered_corpus = {
            key: value
            for key, value in corpus.items()
            if value["title"].strip() != "" or value["text"].strip() != ""
        }

        # Combine the title and text into a single string
        combined_corpus = {
            key: (value["title"] + " " + value["text"]).strip()
            for key, value in filtered_corpus.items()
        }

        # Initialize the base class with the combined corpus
        super().__init__(combined_corpus)


class MiraclCorpusDataset(Dataset):
    def __init__(self, corpus, transform_lambda=None):
        """
        Args:
            corpus (dict): The corpus dictionary where each key is a document ID, and the value is a dictionary with "title" and "text".
        """
        # Combine the title and text into a single string
        logger.info("start parsing miracl corpus")
        self.corpus = corpus
        self.transform_lambda = transform_lambda

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        data = self.corpus[idx]
        key = data["docid"]
        value = data["title"] + " " + data["text"]
        if self.transform_lambda is not None:
            value = self.transform_lambda(value)
        return key, value


class DDPDatasetWithRank(Dataset):
    def __init__(
        self, inner_dataset, local_rank, world_size, drop=False, shuffle=False
    ):
        self.inner_dataset = inner_dataset
        number_of_samples = len(self.inner_dataset)
        if drop:
            number_of_samples = number_of_samples - number_of_samples % world_size
        self.idxs = [
            i for i in range(number_of_samples) if i % world_size == local_rank
        ]
        if shuffle:
            state = random.getstate()
            random.seed(local_rank)
            random.shuffle(self.idxs)
            random.setstate(state)
        logger.info(
            f"local index {local_rank}, world size {world_size}, local sample number {len(self.idxs)}"
        )

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.inner_dataset[self.idxs[idx]]


class KnowledgeDistillDataset(Dataset):
    def __init__(self, all_data, sample_num=2, shuffle=True):
        """
        Args:
            data: A list of dictionary which have "query", "docs" and "scores" list.
                An example: {"query":"hi","docs":["hello","hi"],"scores":[0.99,1]}
        """
        self.data = []
        logger.info(f"KnowledgeDistillDataset input data number: {len(all_data)}")
        assert sample_num >= 2

        has_scores = "scores" in all_data[0]

        for data in all_data:
            query = data["query"]
            texts = data["docs"]
            if has_scores:
                scores = data["scores"]
                assert len(texts) == len(scores)
            else:
                scores = [None] * len(texts)

            if shuffle:
                idxs = list(range(len(scores)))
                random.shuffle(idxs)
                for i in range(0, len(scores), sample_num):
                    # we need to make sure all samples have same negative number
                    # to split all documents and perform matmul
                    if len(scores) - i < sample_num:
                        break
                    sample_idxs = idxs[i : i + sample_num]
                    self.data.append(
                        [
                            query,
                            [texts[idx] for idx in sample_idxs],
                            [scores[idx] for idx in sample_idxs],
                        ]
                    )
            else:
                idxs = list(range(len(scores)))
                step = len(scores) // sample_num
                for i in range(0, step):
                    sample_idxs = [idxs[j * step + i] for j in range(sample_num)]
                    self.data.append(
                        [
                            query,
                            [texts[idx] for idx in sample_idxs],
                            [scores[idx] for idx in sample_idxs],
                        ]
                    )
        logger.info(
            f"KnowledgeDistillDataset after process data number: {len(self.data)}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MsMarcoKDDataset(KnowledgeDistillDataset):
    @staticmethod
    def transform_str(s):
        try:
            return s.encode("latin1").decode("utf-8")
        except:
            return s

    def __init__(self, score_dic_path, corpus=None, queries=None, sample_num=2):
        if corpus is None or queries is None:
            assert score_dic_path is not None
            beir_resource_url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
            local_data_path = util.download_and_unzip(beir_resource_url, "data/beir/")
            corpus, queries, qrels = GenericDataLoader(
                data_folder=local_data_path
            ).load(split="train")

        with open(score_dic_path) as f:
            score_dic = json.load(f)

        for k, v in tqdm(corpus.items()):
            corpus[k]["text"] = MsMarcoKDDataset.transform_str(v["text"])

        all_data = []
        for q_id in tqdm(score_dic.keys()):
            doc_ids = score_dic[q_id]["doc_id"]
            scores = score_dic[q_id]["score"]
            texts = [corpus[doc_id]["text"] for doc_id in doc_ids]
            data = {"query": queries[q_id], "docs": texts, "scores": scores}
            all_data.append(data)

        super().__init__(all_data=all_data, sample_num=sample_num, shuffle=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PosNegsDataset(Dataset):
    def __init__(self, data, sample_num=3, **kwargs):
        """
        Args:
            data: A list of dictionary which have "query", "pos" and "negs" list.
                An example: {"query":"hi","pos":"hello","negs":["hi","world"]}
                "negs" is optional
        """
        assert sample_num >= 1
        self.data = []
        logger.info(f"input data number: {len(data)}")
        for dict_ in tqdm(data):
            dict_["negs"] = dict_["negs"]
            for i in range(0, len(dict_.get("negs", [])), sample_num):
                if len(dict_["negs"]) - i < sample_num:
                    break
                self.data.append(
                    [
                        dict_["query"],
                        dict_["pos"],
                        dict_.get("negs", [])[i : i + sample_num],
                    ]
                )
        logger.info(f"after process data number: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MiraclTrainingDataset(Dataset):
    def __init__(self, lang="en", source="miracl/miracl", dataset=None):
        if dataset is None:
            miracl = load_dataset(source, lang, trust_remote_code=True)
            self.dataset = miracl["train"]
        else:
            self.dataset = dataset

        self.idx_to_data = []
        self.neg_passages = []
        for i, data in tqdm(enumerate(self.dataset)):
            for j in range(len(data["positive_passages"])):
                self.idx_to_data.append((i, j))
            self.neg_passages.append([neg["text"] for neg in data["negative_passages"]])

    def __len__(self):
        return len(self.idx_to_data)

    def __getitem__(self, idx):
        i = self.idx_to_data[idx][0]
        j = self.idx_to_data[idx][1]
        return {
            "query": self.dataset[i]["query"],
            "pos": self.dataset[i]["positive_passages"][j]["text"],
            "negs": self.neg_passages[i],
        }


class CombinedRandomSampler(Sampler):
    def __init__(self, datasets, batch_size, drop_last=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.samplers = [
            BatchSampler(
                RandomSampler(dataset), batch_size=batch_size, drop_last=drop_last
            )
            for dataset in datasets
        ]
        self.dataset_sequences = None

    def set_dataset_sequences(self, dataset_sequences=None):
        if dataset_sequences is None:
            dataset_sequences = [
                [i] * len(self.samplers[i]) for i in range(len(self.samplers))
            ]
            dataset_sequences = list(chain(*dataset_sequences))
            if is_ddp_enabled():
                logger.info("DDP is enabled, set seed to fix dataset sequence iter")

                state = random.getstate()
                random.seed(0)
                random.shuffle(dataset_sequences)
                random.setstate(state)

                logger.info(dataset_sequences[:10])
            else:
                random.shuffle(dataset_sequences)
        self.dataset_sequences = dataset_sequences

    def __iter__(self):
        if self.dataset_sequences is None:
            self.set_dataset_sequences()

        iters = [iter(sampler) for sampler in self.samplers]
        for dataset_idx in self.dataset_sequences:
            idxs = next(iters[dataset_idx])
            yield [[dataset_idx, idx] for idx in idxs]

    def __len__(self):
        return sum(map(len, self.samplers))


class CombinedDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.length = sum(map(len, self.datasets))
        self.no_prepare = True

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dataset_idx, data_idx = idx
        return self.datasets[dataset_idx][data_idx]


DATASET_CLS_MAP = {"kd": KnowledgeDistillDataset, "posnegs": PosNegsDataset}


def load_dataset(path, cls, shuffle=False, sample_num_one_query=2):
    logger.info(f"load dataset from {path}. dataset cls: {DATASET_CLS_MAP[cls]}")
    return DATASET_CLS_MAP[cls](
        DatasetsDataset.load_from_disk(path),
        sample_num=sample_num_one_query,
        shuffle=shuffle,
    )


def load_datasets(path, cls, training_args, shuffle=False, sample_num_one_query=2):
    datasets = []
    for dataset in os.listdir(path):
        dataset_path = os.path.join(path, dataset)
        datasets.append(load_dataset(dataset_path, cls, shuffle, sample_num_one_query))

    datasets = [
        DDPDatasetWithRank(
            dataset,
            training_args.local_process_index,
            training_args.world_size,
            drop=True if training_args.world_size != 1 else False,
            shuffle=True if training_args.world_size != 1 else False,
        )
        for dataset in datasets
    ]
    dataset = CombinedDataset(datasets)

    return dataset
