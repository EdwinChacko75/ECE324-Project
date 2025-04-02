# reward_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from data_loader import load_prm800k

def preprocess_function(examples, tokenizer, max_length):
    """
    Preprocess examples from the PRM800K dataset.
    """
    model_inputs = tokenizer(examples["input"], truncation=True, max_length=max_length, padding="max_length")
    model_inputs["labels"] = [label + 1 for label in examples["label"]]

    return model_inputs

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

def train_reward_model(config):
    model_name = config["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_prm800k(config["dataset"]["path"], split=config["dataset"]["split"])
    
    # Preprocess the dataset; adjust batched size as needed
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer, config["model"]["max_length"]), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=config["training"]["reward_model"]["output_dir"],
        num_train_epochs=config["training"]["reward_model"]["epochs"],
        per_device_train_batch_size=config["training"]["reward_model"]["batch_size"],
        learning_rate=config["training"]["reward_model"]["learning_rate"],
        logging_steps=config["training"]["reward_model"]["logging_steps"],
        save_steps=config["training"]["reward_model"]["save_steps"],
        evaluation_strategy=config["training"]["reward_model"]["evaluation_strategy"],
        fp16=config["training"]["reward_model"]["fp16"],
        bf16=config["training"]["reward_model"]["bf16"],
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    model.save_pretrained(config["training"]["reward_model"]["output_dir"])
    tokenizer.save_pretrained(config["training"]["reward_model"]["output_dir"])
