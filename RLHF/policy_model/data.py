# data.py
import torch
from typing import Dict, Any
from datasets import load_dataset, Dataset
import torch.distributed as dist
from transformers import PreTrainedTokenizerBase
from torch.utils.data import DataLoader, DistributedSampler


def load_rlhf_datasets(
    config: Dict[str, Any], tokenizer: PreTrainedTokenizerBase
) -> DataLoader:
    """
    Loads and tokenizes an RLHF training dataset from a JSON file and returns a DataLoader.

    The function handles:
    - Tokenization using a Hugging Face tokenizer
    - Format conversion to PyTorch tensors
    - Support for distributed training via DistributedSampler

    Args:
        config (dict): Configuration dictionary containing dataset path, model, and training params.
            Must contain:
                - config["dataset"]["train_path"]: Path to the training JSONL file.
                - config["model"]["max_length"]: Max sequence length.
                - config["training"]["rlhf"]["batch_size"]: Batch size.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used to tokenize input text.

    Returns:
        DataLoader: A PyTorch DataLoader with tokenized RLHF training examples.
    """
    dataset_path = config["dataset"]["train_path"]
    dataset: Dataset = load_dataset("json", data_files={"train": dataset_path})["train"]

    def tokenize_fn(examples):
        return tokenizer(
            examples["input"],
            truncation=True,
            max_length=config["model"]["max_length"],
            padding="max_length",
        )

    dataset = dataset.map(tokenize_fn, batched=True, batch_size=1000)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    sampler: DistributedSampler | None = None
    if dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    return DataLoader(
        dataset,
        batch_size=config["training"]["rlhf"]["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
    )
