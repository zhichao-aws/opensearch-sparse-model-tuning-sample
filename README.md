# Exploring $\ell_0$ Sparsicifaction for Inference-free Sparse Retrievers

## Prepare the environment

### Conda environment
Run the following commands to create a conda environment with all the required dependencies:
```
conda env create -f search.yaml
conda activate search
```
or
```
conda create -n search python=3.11
conda activate search
conda install pytorch==2.5.1 cudatoolkit=12.1 -c pytorch
conda install numpy
pip install accelerate==1.1.1 transformers==4.44.1 datasets==3.0.2 opensearch-py beir
```

### OpenSearch service
To evaluate search relevance or mine hard negatives, run an OpenSearch node at local device. It can be accessed at `http://localhost:9200` without username/password(security disabled). For more details, please check [OpenSearch doc](https://opensearch.org/docs/latest/install-and-configure/install-opensearch/tar/). Here are steps to start a node without security:
1. Follow the step1 and step2 in above documentation.
2. Modify `/path/to/opensearch-2.18.0/config/opensearch.yml`, add this line: `plugins.security.disabled: true`
4. (Optional) Modify `/path/to/opensearch-2.18.0/config/jvm.options`, replace: `-Xms1g -Xmx1g` to `-Xms32g -Xmx32g`, if you have enough memory, which is to avoid circuit breaker memory error.
5. Start a tmux session so the OpenSearch won't stop after the terminal is close `tmux new -s opensearch`. In the tmux session, run `cd /path/to/opensearch-2.18.0` and `./bin/opensearch`.
6. The service is running. Run `curl -X GET http://localhost:9200` to test.

### Platform Details
Our result are tested on 8 V100 (16G) GPU and Intel(R) Xeon(R) CPU E5-2686, CUDA 12.2, CentOS.

## Data Preparation

### Download Data
We use the [MS MARCO](https://microsoft.github.io/msmarco/) dataset. Run the following codes for data preparation. 

<!-- Following the previous work, we use the one [sparse](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-v1), one [dense](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) and two cross-encoder model [1](https://huggingface.co/castorini/monot5-3b-msmarco-10k), [2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2) to generate scores on the msmarco dataset. To be specific, for a query, opensearch serives will return the top 100 documents and the model will calculate the query-doc pair score for each query. Each model's output scores will be normalized across batch (50, which means 50*100=5000 query-doc pairs) and take averaged across model to get the final score for each query-dic pair. -->

```bash
torchrun --nproc_per_node=8 data_preparation.py \
  --model_name_or_path opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill
```

Finally, the data should be in the following format:
```
data/msmarco_ft/
├── data-00000-of-00035.arrow
...
dataset_info.json
state.json
```

## Reproduce the results
To reproduce the results, run the following commands:
```
bash run_reproduce.sh
```
This script will run configurations in `config` folder, which are 1) varying lambda 2) varying threshold and 3) varying $\ell_0$ activation. Key parameters we discussed in the paper, here in the configuration files are:

- `flops_d_lambda`: float, the sparsity level of the model.
- `flops_threshold`: int, the threshold for the $\ell_0$ mask loss.
- `activation_type`: str, the activation function type, could be 'relu', 
- `decouple_activation`: bool, whether to decouple the $\ell_0$ activation function.

The results will be saved in the `output` directory based on the configuration in `config` folder.    


## Claim
This codes are built on the codebase of [opensearch-sparse-model-tuning-sample](https://github.com/zhichao-aws/opensearch-sparse-model-tuning-sample). Since the codebase uses the Apache License 2.0, we will also use the same license amd release this codes once the paper is accepted. Currently, this codes is private and only for review purpose.
