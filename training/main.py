import os
import torch
import shutil
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
    DATASET = config["dataset_name"]
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
        base_dir=config["checkpoint_dir"],
        model_name=config.get("run_name_prefix", "run"),
    )
    FINAL_MODEL_DIR = os.path.join(RUN_DIR, "final_model")

    lora_name = "lora" if USE_LORA else "base"
    final_dir =f"{MODEL_NAME[-2:]}_{EPOCHS}_{LEARNING_RATE}_{lora_name}_{DATASET}"
    FINAL_CP_DIR = os.path.join(os.path.dirname(RUN_DIR), final_dir)
    # Load model and tokenizer
    print("Loading model...")
    model, tokenizer = load_model(MODEL_NAME, precision=PRECISION, use_lora=USE_LORA)

    # Load datasets
    print("Preparing datasets...")
    train_dataset = load_and_prepare_dataset(
        tokenizer,
        dataset_name=DATASET,
        split=config["train_split"],
        max_length=MAX_LENGTH,
    )

    eval_dataset = load_and_prepare_dataset(
        tokenizer,
        dataset_name=DATASET,
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
        # early_stopping=config.get("early_stopping", False),
        # repetition_penalty=config.get("repetition_penalty", 1.0),

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
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    if USE_LORA:
        model.save_pretrained(FINAL_MODEL_DIR)
    else:
        torch.save(model.state_dict(), os.path.join(FINAL_MODEL_DIR, "model_weights.pth"))

    # Move out from temporary run directory
    print(f"Training complete. Model saved to: {FINAL_CP_DIR}")
    if os.path.exists(FINAL_CP_DIR):
        shutil.rmtree(FINAL_CP_DIR)  
    shutil.move(RUN_DIR, FINAL_CP_DIR)
    

if __name__ == "__main__":
    main()
