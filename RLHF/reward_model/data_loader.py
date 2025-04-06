# data_loader.py
import os
from datasets import load_dataset

def load_prm800k(config, tokenizer, split="train"):
    """
    Loads the PRM800K dataset from a JSONL file.
    """
    dataset = load_dataset("json", data_files=config["dataset"][f"{split}_path"], split="train")

    dataset = dataset.map(
        lambda x: tokenizer(
            x["input"], truncation=True, max_length=config["model"]["max_length"], padding="max_length"
        ) | {"labels": x["label"]},
        batched=True,
        batch_size=10000,
        load_from_cache_file=config["dataset"]["use_cache"]
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset

def get_output_dir(config):
    return os.path.join(
        config["training"]["reward_model"]["output_dir"],
        f'{config["model"]["base_model"].split("/")[-1]}{"" if not config["model"]["lora"] else "_lora"}'
    )