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

def main():
    parser = argparse.ArgumentParser(description="Calculate MLM loss on the first 10,000 documents of MS MARCO.")
    parser.add_argument("--model_id", type=str, required=True, help="Path or Hugging Face ID of the model")
    parser.add_argument("--tokenizer_id", type=str, required=True, help="Path or Hugging Face ID of the tokenizer")
    
    args = parser.parse_args()

    # 1. Set seed for reproducibility
    set_seed(42)

    # 2. Load Dataset
    print("Loading MS MARCO corpus...")
    # As per hint
    corpus = load_dataset("BeIR/msmarco", "corpus", split="corpus")
    
    # Select first 10,000 documents
    print("Selecting first 2,000 documents...")
    # We use range(2000) to deterministic selection
    dataset = corpus.select(range(2000))

    # 3. Load Tokenizer and Model
    print(f"Loading tokenizer: {args.tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    
    print(f"Loading model: {args.model_id}")
    model = AutoModelForMaskedLM.from_pretrained(args.model_id)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Important: Inference mode

    # 4. Preprocessing
    def preprocess_function(examples):
        # Combine title and text if title exists
        texts = []
        for title, text in zip(examples['title'], examples['text']):
            texts.append(text)
        
        # Tokenize
        # We truncate to 512 (standard BERT length)
        # We do not pad here, we let the DataCollator handle dynamic padding for efficiency
        return tokenizer(texts, truncation=True, max_length=512)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    # 5. Data Collator & DataLoader
    # mlm=True enables masking. mlm_probability defaults to 0.15
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )

    # Batch size 16 is usually safe for 512 seq length on modern GPUs. 
    # If OOM, user might need to adjust, but script requirements didn't specify arg for batch size.
    batch_size = 16
    dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=batch_size, 
        collate_fn=data_collator
    )

    # 6. Calculate Loss
    total_loss = 0.0
    total_batches = 0
    
    print("Calculating MLM loss...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Accumulate loss (outputs.loss is the mean loss for the batch)
            loss = outputs.loss
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    print(f"\nResults:")
    print(f"Processed {len(dataset)} documents.")
    print(f"Average MLM Loss: {avg_loss:.6f}")
    print(f"Perplexity: {torch.exp(torch.tensor(avg_loss)).item():.6f}")

if __name__ == "__main__":
    main()

