# evaluate_reward_model.py
import os
import torch
from typing import Dict, Any
from transformers import Trainer, TrainingArguments
from reward_model.data_loader import load_prm800k
from policy_model.model import load_reward_model, load_tokenizer
from reward_model.trainer import compute_metrics
from utils import load_config


def evaluate_reward_model(
    config: Dict[str, Any], device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, float]:
    """
    Evaluates a trained reward model on the PRM800K test split.

    Args:
        config (dict): Configuration dictionary loaded from YAML.
        device (str): Target device for evaluation (default is CUDA if available).

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Load tokenizer and reward model
    tokenizer = load_tokenizer(config["training"]["rlhf"]["reward_model_dir"])
    model = load_reward_model(config, device)
    model.to(device)
    model.eval()

    # Load the test dataset
    test_dataset = load_prm800k(config, tokenizer, split="test")

    # Torchrun compatibility
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=32,
        dataloader_drop_last=False,
        report_to=[],
        local_rank=local_rank,
    )

    trainer = Trainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics,
    )

    # Evaluate
    results = trainer.evaluate(eval_dataset=test_dataset)

    # Print results on main process only
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    ):
        print("Evaluation Results:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")

    return results


if __name__ == "__main__":
    config = load_config("./config.yaml")
    evaluate_reward_model(config)
