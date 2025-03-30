import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType


model_name = None
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# configure LoRA 
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)

# load GSM8K dataset
dataset = load_dataset("gsm8k", split="train")

# prepare tokenization
def tokenize_function(example):
    prompt = f"Question: {example['question']}\nAnswer: {example['answer']}"
    return tokenizer(prompt, truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# set up training arguments
training_args = TrainingArguments(
    output_dir="./lora_finetuned",
    per_device_train_batch_size=4,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=200,
    num_train_epochs=3,
    logging_steps=50,
    learning_rate=2e-4,
    fp16=True
)

# initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./lora_finetuned_final")
tokenizer.save_pretrained("./lora_finetuned_final")
