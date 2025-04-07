# data_loader.py
import os
from typing import Dict, Any
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


def load_prm800k(
    config: Dict[str, Any], tokenizer: PreTrainedTokenizerBase, split: str = "train"
) -> Dataset:
    """
    Loads and tokenizes the PRM800K dataset from a local JSONL file.

    Args:
        config (dict): Configuration dictionary containing dataset and model details.
        tokenizer (PreTrainedTokenizerBase): Hugging Face tokenizer for input processing.
        split (str): Dataset split to load (e.g., "train", "validation").

    Returns:
        Dataset: A tokenized PyTorch-formatted dataset ready for training.
    """
    dataset = load_dataset(
        "json", data_files=config["dataset"][f"{split}_path"], split="train"
    )

    dataset = dataset.map(
        lambda x: tokenizer(
            x["input"],
            truncation=True,
            max_length=config["model"]["max_length"],
            padding="max_length",
        )
        | {"labels": x["label"]},
        batched=True,
        batch_size=10000,
        load_from_cache_file=config["dataset"]["use_cache"],
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def get_output_dir(config: Dict[str, Any]) -> str:
    """
    Constructs an output directory path based on the base model name and LoRA usage.

    Args:
        config (dict): Configuration dictionary.

    Returns:
        str: Output directory path.
    """
    base_name = config["model"]["base_model"].split("/")[-1]
    suffix = "_lora" if config["model"].get("lora", False) else ""
    return os.path.join(
        config["training"]["reward_model"]["output_dir"],
        f"{base_name}{suffix}",
    )
