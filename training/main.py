import os
import torch
import yaml
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from model import load_model
from dataset import load_and_prepare_dataset
from utils import create_run_directory, load_config
from metrics import compute_metrics




def main():
    # Load configuration
    config = load_config()

    # Basic training settings
    MODEL_NAME = config["model_name"]
    USE_LORA = config.get("use_lora", False)
    BATCH_SIZE = config["batch_size"]
    EVAL_BATCH_SIZE = config.get("eval_batch_size", BATCH_SIZE)
    EPOCHS = config["epochs"]
    LEARNING_RATE = config["learning_rate"]
    MAX_LENGTH = config["max_length"]

    # Precision setting
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    PRECISION = dtype_map.get(config.get("precision", "float16"), torch.float16)

    # Create run directory
    RUN_DIR = create_run_directory(
        base_dir=config["base_checkpoint_dir"],
        model_name=config.get("run_name_prefix", "run"),
    )
    FINAL_MODEL_DIR = os.path.join(RUN_DIR, "final_model")

    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model(MODEL_NAME, precision=PRECISION, use_lora=USE_LORA)

    # Load datasets
    print("Preparing datasets...")
    train_dataset = load_and_prepare_dataset(
        tokenizer,
        dataset_name=config["dataset_name"],
        split=config["train_split"],
        max_length=MAX_LENGTH,
    )

    eval_dataset = load_and_prepare_dataset(
        tokenizer,
        dataset_name=config["dataset_name"],
        split=config["eval_split"],
        max_length=MAX_LENGTH,
    )

    # Set up data collator for Causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=config['masked_language_modelling'])

    # Set up HuggingFace training arguments
    training_args = TrainingArguments(
        output_dir=RUN_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=config.get("weight_decay", 0.0),
        warmup_steps=config.get("warmup_steps", 0),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        fp16=(PRECISION == torch.float16),
        bf16=(PRECISION == torch.bfloat16),
        evaluation_strategy=config.get("evaluation_strategy", "steps"),
        eval_steps=config.get("eval_steps", 100),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 200),
        logging_steps=config.get("logging_steps", 50),
        save_total_limit=config.get("save_total_limit", 2),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        fp16_full_eval=config.get("fp16_full_eval", False),
        load_best_model_at_end=config.get("load_best_model_at_end", False),
    )

    # Initialize HuggingFace Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics(tokenizer),
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save final model
    print("Saving model...")
    model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"Training complete. Model saved to: {FINAL_MODEL_DIR}")


if __name__ == "__main__":
    main()
