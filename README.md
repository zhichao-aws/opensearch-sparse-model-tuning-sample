# Exploring $\ell_0$ Sparsicifaction for Inference-free Sparse Retrievers
<div>
    <p>
        <a href='https://arxiv.org/abs/2504.14839'><img src='https://img.shields.io/badge/arXiv-2504.14839-b31b1b'></a>
        <img src="https://img.shields.io/badge/python-3.11-blue">
        <a href='https://opensearch.org/slack.html'><img src='https://img.shields.io/badge/Slack-Join-green'></a>
        <a href='https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill'><img src='https://img.shields.io/badge/Hugging%20Face-Model%20Weights-blue'></a>
        <a href='https://paperswithcode.com/sota/zero-shot-on-beir-inference-free-model-on?p=exploring-ell-0-sparsification-for-inference'><img src='https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/exploring-ell-0-sparsification-for-inference/zero-shot-on-beir-inference-free-model-on'></a>
    </p>
</div>

## News
+ 📢: We have released our paper on arXiv. Check it out [here](https://arxiv.org/abs/2504.14839).
+ 📢: Our model is now available on Hugging Face. Try it [here](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill).
+ 📢: Our paper has been accepted by SIGIR 2025.

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

## Run Sample Config
To run sample experiment, run the following commands:
```
bash run_sample.sh
```

Key parameters we discussed in the paper, here in the configuration files are:

- `flops_d_lambda`: float, the sparsity level of the model.
- `flops_threshold`: int, the threshold for the $\ell_0$ mask loss.
- `activation_type`: str, the activation function type, could be 'relu', 
- `decouple_activation`: bool, whether to decouple the $\ell_0$ activation function.

The results will be saved in the `output` directory based on the configuration.  

## FAQ
+ **How can I finetune the model on my own?**
  Great! Please check the main branch, which shows how to use our code to finetune the model.
  
+ **What is the difference between the released Hugging Face model and the paper-reported model?**
  We performed a fair zero-shot experiment and reported the performance in our paper. For the model released on Hugging Face, we added more data (detailed in the Hugging Face README) for production purposes.
  
+ **More questions?**
  Please don't hesitate to open an issue or contact the authors.


## Cite
If you find our work helpful, please cite it as follows:
```bibtex
@article{shen2025explore,
  title={Exploring $\ell_0$ Sparsification for Inference-free Sparse Retrievers},
  author={Xinjie Shen and Zhichao Geng and Yang Yang},
  year={2025},
  eprint={2504.14839},
  archivePrefix={arXiv},
  primaryClass={cs.IR}
}

