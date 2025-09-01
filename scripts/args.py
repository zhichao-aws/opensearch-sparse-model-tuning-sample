import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Union

from transformers import HfArgumentParser, TrainingArguments

beir_datasets = "trec-covid,nfcorpus,nq,hotpotqa,fiqa,arguana,webis-touche2020,dbpedia-entity,scidocs,fever,climate-fever,scifact,quora"
miracl_datasets = "bn,te,es,fr,id,hi,ru,ar,zh,fa,ja,fi,sw,ko,en"
tydi_datasets = (
    "arabic,bengali,english,finnish,indonesian,japanese,korean,russian,swahili,telugu"
)
nano_beir_datasets = "NanoClimateFEVER,NanoDBPedia,NanoFEVER,NanoFiQA2018,NanoHotpotQA,NanoNFCorpus,NanoNQ,NanoQuoraRetrieval,NanoSCIDOCS,NanoArguAna,NanoSciFact,NanoTouche2020"


@dataclass
class DataTrainingArguments:
    max_seq_length: int = field(default=512)
    eval_max_seq_length: int = field(default=512)
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
    flops_q_lambda: float = field(default=None)
    flops_q_T: float = field(default=None)
    ranking_loss_weight: float = field(default=1)
    kd_ensemble_teacher_kwargs: Optional[Union[dict, str]] = field(
        default_factory=dict,
    )
    idf_lr: Optional[float] = field(default=None)
    first_rank_thresh: int = field(default=10000)
    use_two_phase: bool = field(default=False)
    skip_ingest: bool = field(default=False)
    do_search: bool = field(default=True)
    query_prune: float = field(default=0)
    flops_threshold: int = field(default=None)
    swap_times: float = field(default=0)
    temperature: float = field(default=1.0)
    score_scale: float = field(default=1.0)

    def __post_init__(self):
        return


@dataclass
class ModelArguments:
    inf_free: bool = field(default=True)
    model_name_or_path: str = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    idf_path: Optional[str] = field(default=None)
    idf_requires_grad: Optional[bool] = field(default=False)
    prune_ratio: Optional[float] = field(default=None)
    preprocess_func: Optional[str] = field(default=None)
    use_l0: bool = field(default=False)

    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name_or_path
        if self.idf_path == "null":
            self.idf_path = None
        if self.preprocess_func == "null":
            self.preprocess_func = None
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
