# dataset.py
import os
from typing import Dict, Any
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase


def prepare_example(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Prepare a single example from the dataset by formatting it into a Chain-of-Thought (CoT) style prompt.

    Args:
        example (dict): A dictionary with keys "question" and "answer".

    Returns:
        dict: A dictionary containing the formatted "prompt" and "full_text".
    """
    question = example.get("question")
    answer = example.get("answer")

    # Create a Chain-of-Thought style prompt that encourages reasoning
    prompt = f"Question: {question}\nLet's think step by step."

    # Combine prompt and answer for full training text
    full_answer = f"{prompt}\n{answer}"  # full input sequence

    return {"prompt": prompt, "full_text": full_answer}


def tokenize_function(
    example: Dict[str, str], tokenizer: PreTrainedTokenizerBase, max_length: int = 512
) -> Dict[str, Any]:
    """
    Tokenize the example's prompt and full text, and prepare labels for training.

    Args:
        example (dict): A dictionary containing "prompt" and "full_text".
        tokenizer: The tokenizer instance to tokenize the text.
        max_length (int): Maximum length of tokenized sequences.

    Returns:
        dict: A dictionary with tokenized input IDs, attention masks, and masked labels.
    """
    full_text = example["full_text"]
    prompt = example["prompt"]

    # Tokenize prompt and full_text with truncation only (no padding)
    full_tokens = tokenizer(full_text, truncation=True, max_length=max_length)
    prompt_tokens = tokenizer(prompt, truncation=True, max_length=max_length)

    # Create labels: copy input_ids, then mask out prompt tokens with -100
    labels = full_tokens["input_ids"].copy()
    prompt_len = len(prompt_tokens["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len

    # Add the labels to the tokenized output
    full_tokens["labels"] = labels

    return full_tokens


def load_and_prepare_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = "gsm8k",
    split: str = "train",
    max_length: int = 512,
) -> Dataset:
    """
    Load a dataset, prepare each example with CoT formatting, and tokenize the dataset.

    Args:
        tokenizer: Tokenizer used for converting text to tokens.
        dataset_name (str): Name of the dataset to load.
        split (str): Which split to load (e.g., "train", "test").
        max_length (int): Maximum token length for input sequences.

    Returns:
        Dataset: A HuggingFace Dataset object with tokenized examples ready for training.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", f"{dataset_name}_{split}.jsonl")

    if os.path.exists(file_path):
        dataset = load_dataset("json", data_files=file_path, split="train")
        print(f"Loading processed data from {file_path}")
    else:
        dataset = load_dataset(dataset_name, "main", split=split)

    dataset = dataset.map(prepare_example, load_from_cache_file=False)

    tokenized_dataset = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    return tokenized_dataset
