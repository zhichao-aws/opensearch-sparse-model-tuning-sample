import sys
import os

from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import HfArgumentParser, TrainingArguments

beir_datasets = "trec-covid,nfcorpus,nq,hotpotqa,fiqa,arguana,webis-touche2020,dbpedia-entity,scidocs,fever,climate-fever,scifact,quora"
miracl_datasets = "bn,te,es,fr,id,hi,ru,ar,zh,fa,ja,fi,sw,ko,en"
tydi_datasets = (
    "arabic,bengali,english,finnish,indonesian,japanese,korean,russian,swahili,telugu"
)


@dataclass
class DataTrainingArguments:
    max_seq_length: int = field(default=512)
    train_file: Optional[str] = field(default=None)
    train_file_dir: Optional[str] = field(default=None)
    data_type: Optional[str] = field(default="kd")
    loss_types: List[str] = field(default_factory=lambda: ["kldiv"])
    beir_dir: str = field(default="data/beir")
    miracl_dir: str = field(default="mdata/miracl_eval")
    # dataset name split by comma
    beir_datasets: str = field(default=beir_datasets)
    miracl_datasets: str = field(default=miracl_datasets)
    sample_num_one_query: int = field(default=2)
    use_in_batch_negatives: bool = field(default=False)
    flops_d_lambda: float = field(default=1e-3)
    flops_d_T: float = field(default=10000)
    flops_threshold: float = field(default=200)
    threshold_type: str = field(default="concurrent")
    threshold_mode: str = field(default="ratio")
    flops_q_lambda: float = field(default=None)
    flops_q_T: float = field(default=None)
    ranking_loss_weight: float = field(default=1)
    kd_ensemble_teacher_kwargs: Optional[Union[dict, str]] = field(
        default_factory=dict,
    )
    idf_lr: Optional[float] = field(default=None)

    def __post_init__(self):
        return


@dataclass
class ModelArguments:
    inf_free: bool = field(default=True)
    model_name_or_path: str = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    idf_path: Optional[str] = field(default=None)
    split_batch: Optional[int] = field(default=1)
    idf_requires_grad: Optional[bool] = field(default=False)

    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path
        return


@dataclass
class MiningArguments:
    mine_datasets: str = field(default=None)
    source: str = field(default=None)


def parse_args():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.makedirs(training_args.output_dir, exist_ok=True)

    return model_args, data_args, training_args
