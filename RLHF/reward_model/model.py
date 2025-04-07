# model.py
from transformers import PreTrainedModel
from typing import Tuple, Dict, Any, Union
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_model(
    config: Dict[str, Any],
) -> Tuple[Union[PreTrainedModel, PeftModel], PreTrainedTokenizerBase]:
    """
    Loads a causal language model with an optional value head and LoRA adapter.

    Args:
        config (dict): Configuration dictionary. Must contain:
            - config["model"]["base_model"]: Hugging Face model identifier.
            - config["model"]["lora"]: (Optional) If True, apply LoRA adapters.

    Returns:
        tuple: (model, tokenizer)
            - model: Either a PreTrainedModel or a PEFT-wrapped model with a value head.
            - tokenizer: The corresponding tokenizer instance.
    """
    model_name = config["model"]["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, num_labels=1)

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


def save_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: Dict[str, Any],
    output_dir: str,
) -> None:
    """
    Saves a model and its tokenizer to the specified output directory.

    If LoRA is used, it merges the adapters into the base model before saving.

    Args:
        model (PreTrainedModel): The model to save (optionally wrapped with LoRA).
        tokenizer (PreTrainedTokenizerBase): The tokenizer to save.
        config (dict): Configuration dict used to check if LoRA was applied.
        output_dir (str): Path to the directory where model/tokenizer will be saved.
    """
    if config["model"].get("lora", False):
        model = model.merge_and_unload()
        model.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir, safe_serialization=False)

    tokenizer.save_pretrained(output_dir, safe_serialization=False)
    print(f"Model and tokenizer saved to: {output_dir}")
