## Download model checkpoints

anonymous link: https://zenodo.org/records/18158860?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjlmNjcwYmFmLTgxMzYtNGVlMi1iZTgwLTk4OGUzNmVjNmYzNSIsImRhdGEiOnt9LCJyYW5kb20iOiIxMGM5MmYwY2JhMjM3MWM5MGNjNWZkYjgzMWJjZDFlNyJ9.-5WvEbpYuLWW7YAAgG-ztNgJu8nhJUWkY5WILXOPvyGm_TIKs8yzmFDEcTCRTw3kMPi6ogJeFBNY-khgxQbNOw

## Prepare the environment

### Conda environment
```
conda create -n neural-sparse-mdbert python=3.9
conda activate neural-sparse-mdbert
pip install -r requirements.txt
```

### OpenSearch service
To evaluate search relevance, run an OpenSearch node at local device. It can be accessed at `http://localhost:9200` without username/password(security disabled). For more details, please check [OpenSearch doc](https://opensearch.org/docs/latest/install-and-configure/install-opensearch/tar/). Here are steps to start a node without security:
1. download opensearch and unzip it 
```
wget https://artifacts.opensearch.org/releases/bundle/opensearch/3.4.0/opensearch-3.4.0-linux-x64.tar.gz
tar -xzf https://artifacts.opensearch.org/releases/bundle/opensearch/3.4.0/opensearch-3.4.0-linux-x64.tar.gz
```
2. Modify `/path/to/opensearch-3.4.0/config/opensearch.yml`, add this line: `plugins.security.disabled: true`
3. Start a tmux session so the OpenSearch won't stop after the terminal is close `tmux new -s opensearch`. In the tmux session, run `cd /path/to/opensearch-3.4.0` and `./bin/opensearch`.
4. The sevice is running. Run `curl -X GET http://localhost:9200` to test.

## Run vocab transfer

### VT: Embedding and Bias Initialization

```
cd vocab_trans
python transform.py --save_path mdbert-vt --set_bias --save_additional_tokens
cd ..
```

### run VT training

1. prepare data for MLM (wikibook). We used script from an exising repo https://github.com/ma787639046/bowdpr/tree/main/examples/pretrain. 
```
python prepare_pretrain_data.py --save_to /path/to/this_repo/data/wikibook.ml128.jsonl --maxlen 128
```

2. prepare data for LSR
```
mkdir .cache
cd .cache
wget "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz?download=true" -O cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz
cd ..
```

2. run training. It trains model with MLM task, then run LSR training
```
bash run_vt.sh
```