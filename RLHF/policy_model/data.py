import torch
from datasets import load_dataset

def load_rlhf_datasets(config, tokenizer):
    dataset_path = config["dataset"]["train_path"]
    dataset = load_dataset("json", data_files={"train": dataset_path})["train"]

    def tokenize_fn(examples):
        return tokenizer(
            examples["input"],
            truncation=True,
            max_length=config["model"]["max_length"],
            padding="max_length"
        )

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["rlhf"]["batch_size"],
        shuffle=True
    )

