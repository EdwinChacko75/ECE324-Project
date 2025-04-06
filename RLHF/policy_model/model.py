import os
import json
import torch.distributed as dist
from transformers import GenerationConfig

from safetensors.torch import load_file
from torch.nn import DataParallel
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP

def load_tokenizer(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

def load_policy_model(config, device):
    policy_model_path = config["training"]["rlhf"]["policy_model_dir"]
    
    # Load model with checkpointing enabled
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

    # Enable gradient checkpointing after PEFT wrapping
    if config["training"]["rlhf"]["grad_checkpointing"]:
        print("Using Gradient Checkpointing.")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Move to device
    model = model.to(device)

    # Set generation_config so PEFT won't crash
    gen_cfg = GenerationConfig(
        max_new_tokens=config["training"]["rlhf"].get("max_new_tokens", 50),
        do_sample=True,
        temperature=1.0,
    )
    gen_cfg.max_length = None # To stop being spammed with errors.
    try:
        model.base_model.model.generation_config = gen_cfg  # LoRA-wrapped
    except AttributeError:
        model.generation_config = gen_cfg  # Non-wrapped


    # Wrap in DDP or fallback to DataParallel
    if dist.is_initialized():
        model = DDP(model, device_ids=[device.index])
    else:
        model = DataParallel(model)

    return model

def load_reward_model(config, device):
    path = config["training"]["rlhf"]["policy_model_dir"]

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=True
    )

    # Wrap with value head
    model = AutoModelForCausalLMWithValueHead(base_model)

    # Load state dict from safetensors
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

