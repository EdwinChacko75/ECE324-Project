import os
import torch
from transformers import Trainer, TrainingArguments
from reward_model.data_loader import load_prm800k
from policy_model.model import load_reward_model, load_tokenizer
from reward_model.trainer import compute_metrics
from utils import load_config

def evaluate_reward_model(config, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load model and tokenizer
    tokenizer = load_tokenizer(config["training"]["rlhf"]["reward_model_dir"])
    model= load_reward_model(config, device)
    model.to(device)
    model.eval()

    # Load test dataset
    test_dataset = load_prm800k(config, tokenizer, split="test")

    # Support for torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=32,
        dataloader_drop_last=False,
        report_to=[],
        local_rank=local_rank,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics,
    )

    # Run evaluation
    results = trainer.evaluate(eval_dataset=test_dataset)

    # Only print on rank 0
    if torch.distributed.get_rank() == 0 or not torch.distributed.is_initialized():
        print("Evaluation Results:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")

    return results


if __name__== "__main__":
    config = load_config('./config.yaml')
    evaluate_reward_model(config)
