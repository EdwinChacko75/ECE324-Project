# reward_model.py
from transformers import TrainingArguments
from .data_loader import load_prm800k, get_output_dir
from .trainer import RewardTrainer, compute_metrics
from .model import load_model, save_model


def train_reward_model(config):
    reward_model = config["training"]["reward_model"]

    output_dir = get_output_dir(config)

    # Load model and tokenizer
    model, tokenizer = load_model(config)

    # Load datasetss
    train_dataset = load_prm800k(config, tokenizer, split=config["dataset"]["split"])
    test_dataset = load_prm800k(config, tokenizer, split="test")

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=reward_model["epochs"],
        per_device_train_batch_size=reward_model["batch_size"],
        learning_rate=reward_model["learning_rate"],
        logging_steps=reward_model["logging_steps"],
        save_steps=reward_model["save_steps"],
        eval_strategy=reward_model["eval_strategy"],
        save_strategy=reward_model["save_strategy"],
        eval_steps=reward_model["metric_steps"],
        fp16=reward_model["fp16"],
        bf16=reward_model["bf16"],
        remove_unused_columns=False,
        save_safetensors=False,
        label_names=[],
        max_grad_norm=1.0,
        ddp_find_unused_parameters=False,
    )

    # Custom Trainer
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(f"Reward model training complete.")

    # Save model and tokenizer
    save_model(model, tokenizer, config, output_dir)
