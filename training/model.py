import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

def load_model(MODEL_NAME, precision=torch.float16, use_lora=False):
    """
    Loads the model and tokenizer. Optionally applies LoRA.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # LLaMA models may not have a pad token so set it to the eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=precision, device_map="auto"
    )

    if use_lora:
        # Configure LoRA parameters
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        model = get_peft_model(model, lora_config)
        print("LoRA is enabled and has been applied to the model.")
    else:
        print("Using the base model without LoRA.")

    return model, tokenizer
