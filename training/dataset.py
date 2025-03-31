from datasets import load_dataset


def prepare_example(example):
    """
    Prepare an example by constructing a prompt.
    Modify this function to include chain-of-thought details if desired.
    """
    question = example.get("question")
    answer = example.get("answer")
    # Customize your prompt formatting here.
    prompt = f"Question: {question}\nAnswer: {answer}"
    return {"prompt": prompt}

def tokenize_function(example, tokenizer, max_length=512):
    """
    Tokenizes the prompt using the provided tokenizer.
    """
    # Use chain-of-thought formatting if available; for now, simply tokenize the prompt.
    tokenized = tokenizer(example["prompt"], truncation=True, max_length=max_length)
    return tokenized

def load_and_prepare_dataset(tokenizer, dataset_name="gsm8k", split="train", max_length=512):
    """
    Load the dataset and apply the preparation and tokenization.
    """
    dataset = load_dataset(dataset_name, "main", split=split)
    dataset = dataset.map(prepare_example, load_from_cache_file=False)
    tokenized_dataset = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    return tokenized_dataset
