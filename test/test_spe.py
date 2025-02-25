import os
import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from transformers import get_linear_schedule_with_warmup
import json

from ..scripts.model.sparse_encoders import SparseModel


class DummyDataset(Dataset):
    def __init__(self, size=1000, seq_length=256):
        self.size = size
        self.seq_length = seq_length

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, 32100, (self.seq_length,))
        attention_mask = torch.ones_like(input_ids)
        # Create target vectors (assuming binary classification for simplicity)
        target = torch.randint(0, 2, (32100,)).float()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": target,
        }


class CustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(outputs)
        return self.bce(probs, targets)


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def main():
    # Setup DDP
    local_rank = setup_ddp()

    # Load config
    config = {
        "model_name_or_path": "google/flan-t5-small",
        "model_type": "flan-t5",
        "max_seq_length": 256,
        "per_device_train_batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 500,
        "max_steps": 1000,
        "logging_steps": 50,
        "save_steps": 500,
        "output_dir": "output",
        "inf_free": True,
    }

    # Create model
    model = SparseModel(
        model_id=config["model_name_or_path"],
        model_type=config["model_type"],
        activation_type="relu",
    )

    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    # Move model to GPU and wrap with DDP
    model = model.cuda()
    model = DDP(model, device_ids=[local_rank])

    # Create loss function
    criterion = CustomLoss().cuda()

    # Create dataset and dataloader
    dataset = DummyDataset(size=1000, seq_length=config["max_seq_length"])

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=config["per_device_train_batch_size"],
        sampler=sampler,
        drop_last=True,
    )

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["max_steps"],
    )

    # Training loop
    model.train()
    global_step = 0

    while global_step < config["max_steps"]:
        for batch in dataloader:
            # Move batch to GPU
            batch = {k: v.cuda() for k, v in batch.items()}

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass with gradient computation
            outputs = model(
                inf_free=config["inf_free"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            # Ensure outputs require gradients
            if not outputs.requires_grad:
                outputs = outputs.detach().requires_grad_(True)

            # Compute loss
            loss = criterion(outputs, batch["target"])

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
            scheduler.step()

            global_step += 1

            if local_rank == 0 and global_step % config["logging_steps"] == 0:
                print(f"Step {global_step}, Loss: {loss.item():.4f}")

                # # Print gradient information for debugging
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         print(f"{name}: grad_exists={param.grad is not None}")

            if global_step >= config["max_steps"]:
                break

    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    # move the idf.json file to the current directory
    main()
