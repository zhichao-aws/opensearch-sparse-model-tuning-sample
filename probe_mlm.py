import argparse
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_sparse_rep(logits, attention_mask):
    # Imitating SparseModel._encode from scripts/model/sparse_encoders.py
    # logits: (batch, seq_len, vocab_size)
    # attention_mask: (batch, seq_len)
    
    # values, _ = torch.max(output * kwargs.get("attention_mask").unsqueeze(-1), dim=1)
    values, _ = torch.max(logits * attention_mask.unsqueeze(-1), dim=1)
    
    # values = torch.log1p(torch.relu(values))
    values = torch.log1p(torch.relu(values))
    
    return values

def main():
    parser = argparse.ArgumentParser(description="Calculate MLM loss on the first 2,000 documents of MS MARCO.")
    parser.add_argument("--model_id", type=str, required=True, help="Path or Hugging Face ID of the model")
    parser.add_argument("--tokenizer_id", type=str, required=True, help="Path or Hugging Face ID of the tokenizer")
    
    args = parser.parse_args()

    # 1. Set seed for reproducibility
    set_seed(42)

    # 2. Load Dataset
    print("Loading MS MARCO corpus...")
    corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    
    # Select first 2,000 documents
    print("Selecting first 2,000 documents...")
    dataset = corpus.select(range(2000))

    # 3. Load Tokenizer and Model
    print(f"Loading tokenizer: {args.tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    
    print(f"Loading model: {args.model_id}")
    model = AutoModelForMaskedLM.from_pretrained(args.model_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 4. Preprocessing
    def preprocess_function(examples):
        texts = []
        for title, text in zip(examples['title'], examples['text']):
            # User modification: only use text
            texts.append(text)
        
        return tokenizer(texts, truncation=True, max_length=512)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    # 5. Data Collator & DataLoader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )

    batch_size = 16
    dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )

    # 6. Calculate Loss and Stats
    total_loss = 0.0
    total_batches = 0
    
    # Stats accumulators
    total_flops = 0.0
    total_doc_len = 0.0
    total_nonzero_mean = 0.0
    max_nonzero_global = 0.0
    nonzero_batches_count = 0
    
    # Logits stats accumulators
    total_logits_mean = 0.0
    total_logits_std = 0.0
    
    # Input ID token weight stats
    total_input_token_weight_mean = 0.0
    input_token_weight_batches_count = 0

    print("Calculating MLM loss and Stats...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Calculate stats
            # Logits stats
            logits = outputs.logits
            total_logits_mean += torch.mean(logits).item()
            total_logits_std += torch.std(logits).item()

            # 1. Get Sparse Representation (d_rep)
            # Need attention mask (which is present in batch)
            d_rep = get_sparse_rep(outputs.logits, batch['attention_mask'])
            
            # 2. Calculate FLOPS (d_flops)
            # trainer.py: torch.sum(torch.mean(torch.abs(representation), dim=0) ** 2)
            d_flops = torch.sum(torch.mean(torch.abs(d_rep), dim=0) ** 2).item()
            total_flops += d_flops
            
            # 3. Calculate Avg Doc Length (d_avg_len)
            # trainer.py: (d_rep > 0).sum() / d_rep.shape[0]
            d_avg_len = (d_rep > 0).float().sum(dim=1).mean().item()
            total_doc_len += d_avg_len
            
            # 4. Nonzero entries stats
            # trainer.py: nonzero = d_rep[d_rep > 0]; mean(nonzero), max(nonzero)
            nonzero = d_rep[d_rep > 0]
            if nonzero.numel() > 0:
                total_nonzero_mean += torch.mean(nonzero).item()
                current_max = torch.max(nonzero).item()
                if current_max > max_nonzero_global:
                    max_nonzero_global = current_max
                nonzero_batches_count += 1

            # 5. Input ID token weights stats
            # d_rep shape: (batch_size, vocab_size) ? No, get_sparse_rep returns (batch, vocab) ???
            # Wait, let's check get_sparse_rep implementation carefully.
            # get_sparse_rep: 
            #   values, _ = torch.max(logits * attention_mask.unsqueeze(-1), dim=1) -> (batch, vocab)
            # So d_rep is (batch, vocab_size), representing the max weight for each token in the vocab across the sequence.
            
            # BUT, the user asks for: "每个位置上（不止被mask的），原始的input id对应的token的平均权重"
            # This means we need the weight of the specific token present at each position in the sequence.
            # The sparse model logic (SparseModel._encode) computes a single vector per document (max pooling over sequence).
            # However, `values` before max pooling in SparseModel._encode is:
            # output = self.backbone(**kwargs)[0]  -> (batch, seq_len, vocab)
            # values, _ = torch.max(output * kwargs.get("attention_mask").unsqueeze(-1), dim=1)
            
            # To get weight at each position, we need the logits at that position for the specific input_id.
            # outputs.logits: (batch, seq_len, vocab_size)
            # batch['input_ids']: (batch, seq_len)
            
            # We want to gather the logit value corresponding to the input_id at each position.
            input_ids = batch['input_ids'] # (batch, seq_len)
            # We should only consider non-padding tokens? Yes, usually.
            # And maybe exclude special tokens if we want "original input id"? 
            # Let's stick to attention_mask to exclude padding.
            
            # Gather logits for the input_ids
            # logits.gather(2, input_ids.unsqueeze(-1)) -> (batch, seq_len, 1)
            token_logits = logits.gather(2, input_ids.unsqueeze(-1)).squeeze(-1) # (batch, seq_len)
            
            # Apply the same transformation as sparse model? 
            # User requested raw logits average for input tokens, not processed weights.
            token_weights = token_logits
            
            # Mask out padding
            attention_mask = batch['attention_mask']
            active_weights = token_weights[attention_mask == 1]
            
            if active_weights.numel() > 0:
                total_input_token_weight_mean += torch.mean(active_weights).item()
                input_token_weight_batches_count += 1
            
            total_batches += 1

    avg_loss = total_loss / total_batches
    avg_flops = total_flops / total_batches
    avg_doc_len = total_doc_len / total_batches
    avg_nonzero_value = total_nonzero_mean / nonzero_batches_count if nonzero_batches_count > 0 else 0.0
    avg_logits_mean = total_logits_mean / total_batches
    avg_logits_std = total_logits_std / total_batches
    avg_input_token_weight = total_input_token_weight_mean / input_token_weight_batches_count if input_token_weight_batches_count > 0 else 0.0
    
    print(f"\nResults:")
    print(f"Processed {len(dataset)} documents.")
    print(f"Average MLM Loss: {avg_loss:.6f}")
    print(f"Perplexity: {torch.exp(torch.tensor(avg_loss)).item():.6f}")
    print(f"Average FLOPS (d_flops): {avg_flops:.6f}")
    print(f"Average Doc Length (sparsity): {avg_doc_len:.6f}")
    print(f"Average Nonzero Entry Value: {avg_nonzero_value:.6f}")
    print(f"Max Nonzero Entry Value: {max_nonzero_global:.6f}")
    print(f"Average Logits Mean: {avg_logits_mean:.6f}")
    print(f"Average Logits Std: {avg_logits_std:.6f}")
    print(f"Average Input Token Weight: {avg_input_token_weight:.6f}")

if __name__ == "__main__":
    main()

# python probe_mlm.py --tokenizer_id bert-base-uncased --model_id pretrain/bert-vocab-sb-tn-pmi--20000/checkpoint-20000

# python probe_mlm.py --tokenizer_id bert-base-uncased --model_id pretrain/bert-vocab-sb-tn-pmi--10000/checkpoint-10000

# python probe_mlm.py --tokenizer_id answerdotai/modernbert-base --model_id answerdotai/modernbert-base 
# python probe_mlm.py --tokenizer_id alibaba-nlp/gte-en-mlm-base --model_id alibaba-nlp/gte-en-mlm-base
# python probe_mlm.py --tokenizer_id luyu/co-condenser-marco --model_id luyu/co-condenser-marco