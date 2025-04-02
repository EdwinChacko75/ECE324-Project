# data_loader.py
from datasets import load_dataset

def load_prm800k(path, split="train"):
    """
    Loads the PRM800K dataset from a JSONL file.
    """
    dataset = load_dataset("json", data_files=path, split=split)
    return dataset
