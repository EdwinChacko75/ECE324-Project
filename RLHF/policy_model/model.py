# model.py
import os
import json
import torch
import torch.distributed as dist
from typing import Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from safetensors.torch import load_file
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model


def load_tokenizer(base_model: str) -> PreTrainedTokenizerBase:
    """
    Loads and configures the tokenizer.

    Args:
        base_model (str): Hugging Face model identifier or local path.

    Returns:
        PreTrainedTokenizerBase: Configured tokenizer with padding set.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_policy_model(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """
    Loads a policy model with optional LoRA, gradient checkpointing, and DDP support.

    Args:
        config (dict): Configuration dict with training and model options.
        device (torch.device): Device to move the model to.

    Returns:
        torch.nn.Module: The fully loaded and wrapped policy model.
    """
    policy_model_path = config["training"]["rlhf"]["policy_model_dir"]

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        policy_model_path,
        num_labels=1,
    )

    # Apply LoRA if specified
    if config["training"]["rlhf"].get("lora", False):
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print("Added LoRA Adapters to model.")

    # Enable gradient checkpointing
    if config["training"]["rlhf"]["grad_checkpointing"]:
        print("Using Gradient Checkpointing.")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model = model.to(device)

    # Set up generation config to avoid PEFT crashes
    gen_cfg = GenerationConfig(
        max_new_tokens=config["training"]["rlhf"].get("max_new_tokens", 50),
        do_sample=True,
        temperature=1.0,
    )
    gen_cfg.max_length = None  # Suppress default max length warnings

    try:
        model.base_model.model.generation_config = gen_cfg  # For PEFT-wrapped models
    except AttributeError:
        model.generation_config = gen_cfg  # For non-wrapped models

    # DDP support
    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index])
    else:
        model = DataParallel(model)

    return model


def load_reward_model(config: Dict[str, Any], device: torch.device) -> PreTrainedModel:
    """
    Loads a reward model from a PEFT-trained model directory with v_head weights in safetensors.

    Args:
        config (dict): Configuration containing model path under `training.rlhf.policy_model_dir`.
        device (torch.device): Target device.

    Returns:
        PreTrainedModel: A model with a value head loaded with reward weights.
    """
    path = config["training"]["rlhf"]["policy_model_dir"]

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=True,
    )

    # Wrap with value head
    model = AutoModelForCausalLMWithValueHead(base_model)

    # Load safetensor v_head weights
    index_path = os.path.join(path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError("model.safetensors.index.json not found")

    with open(index_path, "r") as f:
        index = json.load(f)

    v_head_state = {}
    for key, filename in index["weight_map"].items():
        if key.startswith("v_head."):
            safetensor_path = os.path.join(path, filename)
            tensor_file = load_file(safetensor_path)
            for k in tensor_file:
                if k.startswith("v_head."):
                    v_head_state[k.replace("v_head.", "")] = tensor_file[k]

    model.v_head.load_state_dict(v_head_state, strict=False)
    model.is_peft_model = False

    return model.to(device).eval()
