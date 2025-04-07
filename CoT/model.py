# model.py
import torch
from typing import Tuple, Union
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    PreTrainedTokenizer, 
    PreTrainedModel
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    PeftModel
)

def load_model(
    MODEL_NAME: str,
    precision: torch.dtype = torch.float16,
    use_lora: bool = False
) -> Tuple[Union[PreTrainedModel, PeftModel], PreTrainedTokenizer]:
    """
    Loads a pretrained causal language model and its tokenizer. Optionally applies LoRA (Low-Rank Adaptation)
    using the PEFT library for parameter-efficient fine-tuning.

    Args:
        MODEL_NAME (str): Hugging Face model identifier or path to the saved model directory.
        precision (torch.dtype, optional): The desired precision for model weights. Defaults to torch.float16.
        use_lora (bool, optional): Whether to apply LoRA on top of the base model. Defaults to False.

    Returns:
        tuple: (PreTrainedModel, PreTrainedTokenizer)
            The loaded model (optionally wrapped with LoRA) and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=precision,
    )

    if use_lora:
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
