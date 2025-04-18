# model.py
import torch
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


def load_model(
    MODEL_PATH: str, PRECISION: torch.dtype
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    """
    Load a merged model (with or without LoRA), saved via model.save_pretrained(...).

    Args:
        MODEL_PATH (str): Path to the saved model directory.
        PRECISION (torch.dtype): Precision to load the model in.

    Returns:
        tuple (AutoModelForCausalLM, AutoTokenizer): The loaded model and tokenizer.
    """
    print(f"Loading model from {MODEL_PATH} with dtype {PRECISION}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=PRECISION, device_map="auto"
    )

    print("Model and tokenizer successfully loaded.")
    return model, tokenizer
