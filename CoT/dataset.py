from datasets import load_dataset
import os

def prepare_example(example):
    question = example.get("question")
    answer = example.get("answer")
    
    # Create a CoT-style prompt
    prompt = f"Question: {question}\nLet's think step by step."
    full_answer = f"{prompt}\n{answer}"  # full input sequence
    
    return {
        "prompt": prompt,
        "full_text": full_answer
    }

def tokenize_function(example, tokenizer, max_length=512):
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

    full_tokens["labels"] = labels

    return full_tokens



def load_and_prepare_dataset(tokenizer, dataset_name="gsm8k", split="train", max_length=512):
    """
    Load the dataset and apply the preparation and tokenization.
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
