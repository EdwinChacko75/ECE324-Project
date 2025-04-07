from datasets import load_dataset
import os

def prepare_example(example):
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
    
    return {
        "prompt": prompt,
        "full_text": full_answer
    }

def tokenize_function(example, tokenizer, max_length=512):
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

    # Append EOS token if there's room
    # if len(full_tokens["input_ids"]) < max_length:
    #     full_tokens["input_ids"].append(tokenizer.eos_token_id)
    #     full_tokens["attention_mask"].append(1)

    # Create labels: copy input_ids, then mask out prompt tokens with -100
    labels = full_tokens["input_ids"].copy()
    prompt_len = len(prompt_tokens["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len

    # Add the labels to the tokenized output
    full_tokens["labels"] = labels

    return full_tokens



def load_and_prepare_dataset(tokenizer, dataset_name="gsm8k", split="train", max_length=512):
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
    
    # Construct file path to check if a local JSONL file exists
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "data", f"{dataset_name}_{split}.jsonl")

    if os.path.exists(file_path):
        # Load dataset from a local JSON file if it exists
        dataset = load_dataset("json", data_files=file_path, split="train")
        print(f"Loading processed data from {file_path}")
    else:
        # Otherwise, load dataset from HuggingFace Hub
        dataset = load_dataset(dataset_name, "main", split=split)
        
    # Format each example into prompt + full_text format
    dataset = dataset.map(prepare_example, load_from_cache_file=False)
    
    # Tokenize the formatted dataset and remove original columns
    tokenized_dataset = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    return tokenized_dataset

