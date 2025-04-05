import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model, PeftModel
from data_loader import load_prm800k


class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"].float()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        values = outputs[2]

        seq_lens = attention_mask.sum(dim=1) - 1
        last_values = values.squeeze(-1)[torch.arange(values.size(0), device=values.device), seq_lens]

        loss = torch.nn.functional.mse_loss(last_values, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"].float()

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            values = outputs[2]
            seq_lens = attention_mask.sum(dim=1) - 1
            last_values = values.squeeze(-1)[torch.arange(values.size(0), device=values.device), seq_lens]

        return None, last_values.detach().cpu(), labels.detach().cpu()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if predictions.ndim == 3:
        predictions = predictions.squeeze(-1)
    if predictions.ndim == 2:
        predictions = predictions[np.arange(predictions.shape[0]), -1]
    labels = labels.flatten()
    preds_binary = (predictions >= 0.5).astype(int)
    labels_binary = (labels >= 0.5).astype(int)
    accuracy = (preds_binary == labels_binary).mean()
    mse = ((predictions - labels) ** 2).mean()
    return {
        "accuracy": accuracy,
        "mse": mse,
    }


def train_reward_model(config):
    output_dir = os.path.join(
        config["training"]["reward_model"]["output_dir"],
        f'{config["model"]["base_model"].split("/")[-1]}{"" if not config["model"]["lora"] else "_lora"}'
    )
    model_name = config["model"]["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, num_labels=1)

    # Apply LoRA if enabled
    if config["model"].get("lora", False):
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.config.pad_token_id = tokenizer.pad_token_id

    # Load dataset
    train_dataset = load_prm800k(config, tokenizer, split=config["dataset"]["split"])
    test_dataset = load_prm800k(config, tokenizer, split="test")
    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["training"]["reward_model"]["epochs"],
        per_device_train_batch_size=config["training"]["reward_model"]["batch_size"],
        learning_rate=config["training"]["reward_model"]["learning_rate"],
        logging_steps=config["training"]["reward_model"]["logging_steps"],
        save_steps=config["training"]["reward_model"]["save_steps"],
        evaluation_strategy=config["training"]["reward_model"]["evaluation_strategy"],
        save_strategy=config["training"]["reward_model"]["save_strategy"],
        eval_steps=config["training"]["reward_model"]["metric_steps"],
        fp16=config["training"]["reward_model"]["fp16"],
        bf16=config["training"]["reward_model"]["bf16"],
        remove_unused_columns=False,
        save_safetensors=False,
        label_names=[],
        max_grad_norm=1.0,
    )

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save LoRA adapter or full model depending on config
    if config["model"].get("lora", False):
        model = model.merge_and_unload()
        model.save_pretrained(output_dir)  
    else:
        model.save_pretrained(output_dir, safe_serialization=False)

    tokenizer.save_pretrained(output_dir, safe_serialization=False)
