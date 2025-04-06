from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model

def load_model(config):
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
    return model, tokenizer

def save_model(model, tokenizer, config, output_dir):
    if config["model"].get("lora", False):
        model = model.merge_and_unload()
        model.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir, safe_serialization=False)

    tokenizer.save_pretrained(output_dir, safe_serialization=False)
    print(f"Model and tokenizer saved to: {output_dir}")
