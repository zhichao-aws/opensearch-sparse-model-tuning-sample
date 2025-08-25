import datasets

# 1) Load datasets
msmarco_hard_negatives = datasets.load_dataset(
    "opensearch-project/msmarco-hard-negatives", split="train"
)
msmarco_queries = datasets.load_dataset("BeIR/msmarco", "queries")["queries"]
msmarco_corpus = datasets.load_dataset("BeIR/msmarco", "corpus")["corpus"]


# 2) fix occasional text encoding issues
def transform_str(s):
    try:
        s = s.encode("latin1").decode("utf-8")
        return s
    except Exception:
        return s


msmarco_corpus = msmarco_corpus.map(
    lambda x: {"text": transform_str(x["text"])}, num_proc=30
)

# 3) Build convenient lookup tables
id_to_text = {
    _id: text for _id, text in zip(msmarco_corpus["_id"], msmarco_corpus["text"])
}
qid_to_text = {
    _id: text for _id, text in zip(msmarco_queries["_id"], msmarco_queries["text"])
}

# 4) Replace IDs with raw texts to get a text-only dataset
msmarco_hard_negatives = msmarco_hard_negatives.map(
    lambda x: {
        "query": qid_to_text[x["query"]],
        "docs": [id_to_text[doc] for doc in x["docs"]],
    },
    num_proc=30,
)

# 5) Save to disk (directory will contain the text-only view)
msmarco_hard_negatives.save_to_disk("data/msmarco_ft")
