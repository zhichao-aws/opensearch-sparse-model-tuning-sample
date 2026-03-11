import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input jsonl file path")
parser.add_argument("--output", type=str, required=True, help="Output json file path")
parser.add_argument("--max_doc", type=int, default=4, help="Max number of negative docs per query")
parser.add_argument("--model", type=str, default="minicpm", help="Model to use (minicpm, gemma2, or any HuggingFace model path)")
parser.add_argument("--model_type", type=str, default="auto", choices=["auto", "reranker", "sentence_transformer", "bge_m3"],
                    help="Model type: auto (detect), reranker (LLM rerankers), sentence_transformer (bi-encoder), or bge_m3")
args = parser.parse_args()

# Determine model type
model_type = args.model_type
if model_type == "auto":
    if args.model in ["minicpm", "gemma2"]:
        model_type = "reranker"
    elif "bge-m3" in args.model.lower():
        model_type = "bge_m3"
    elif "reranker" in args.model.lower():
        model_type = "reranker"
    else:
        model_type = "sentence_transformer"

print(f"Using model: {args.model}")
print(f"Model type: {model_type}")

# Load model based on type (import only what's needed)
if model_type == "reranker":
    from FlagEmbedding import LayerWiseFlagLLMReranker, LightWeightFlagLLMReranker

    if args.model == "minicpm":
        model = LayerWiseFlagLLMReranker('BAAI/bge-reranker-v2-minicpm-layerwise', use_fp16=True)
    elif args.model == "gemma2":
        model = LightWeightFlagLLMReranker('BAAI/bge-reranker-v2.5-gemma2-lightweight', use_fp16=True)
    else:
        # Try to load as a custom reranker
        try:
            model = LayerWiseFlagLLMReranker(args.model, use_fp16=True)
        except:
            model = LightWeightFlagLLMReranker(args.model, use_fp16=True)
elif model_type == "bge_m3":
    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(args.model, use_fp16=True)
else:
    # sentence_transformer
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

all_pairs = []
all_queries = []
all_docs_list = []
with open(args.input, "r") as f:
    for line in f:
        data = json.loads(line)
        query = data["anchor"]
        docs = [data["positive"]] + data["negatives"][:args.max_doc]
        all_queries.append(query)
        all_docs_list.append(docs)
        for doc in docs:
            all_pairs.append([query, doc])

print(f"Total pairs: {len(all_pairs)}")

# Compute scores based on model type
if model_type == "reranker":
    if args.model == "minicpm":
        scores = model.compute_score(all_pairs, cutoff_layers=[28])
    elif args.model == "gemma2":
        scores = model.compute_score(all_pairs, cutoff_layers=[28], compress_ratio=2, compress_layer=[24, 40])
    else:
        # Custom reranker - try with cutoff_layers
        try:
            scores = model.compute_score(all_pairs, cutoff_layers=[28])
        except:
            scores = model.compute_score(all_pairs)
elif model_type == "bge_m3":
    # BGE-M3 uses dense embeddings for scoring
    queries_text = [pair[0] for pair in all_pairs]
    docs_text = [pair[1] for pair in all_pairs]

    # Encode in batches
    query_embeddings = model.encode(queries_text, batch_size=32)['dense_vecs']
    doc_embeddings = model.encode(docs_text, batch_size=32)['dense_vecs']

    # Compute cosine similarity
    scores = (query_embeddings * doc_embeddings).sum(1).tolist()
else:
    # sentence_transformer - compute embeddings and cosine similarity
    queries_text = [pair[0] for pair in all_pairs]
    docs_text = [pair[1] for pair in all_pairs]

    # Encode in batches
    query_embeddings = model.encode(queries_text, batch_size=32, normalize_embeddings=True)
    doc_embeddings = model.encode(docs_text, batch_size=32, normalize_embeddings=True)

    # Compute cosine similarity (already normalized, so just dot product)
    scores = (query_embeddings * doc_embeddings).sum(1).tolist()

# 后处理：将 scores 按 query 分组，组装成 all_samples
all_samples = []
idx = 0
for query, docs in zip(all_queries, all_docs_list):
    all_samples.append({
        "query": query,
        "docs": docs,
        "scores": scores[idx:idx+len(docs)]
    })
    idx += len(docs)

with open(args.output, "w") as f:
    json.dump(all_samples, f)
