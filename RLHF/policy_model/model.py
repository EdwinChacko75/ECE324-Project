from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model

def load_tokenizer(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

def load_policy_model(config, device):
    base_model = config["model"]["base_model"]
    model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model, num_labels=1)

    if config["training"]["rlhf"].get("use_lora", False):
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    return model.to(device)

def load_reward_model(config, device):
    reward_model_path = config["training"]["rlhf"]["reward_model_dir"]
    model = AutoModelForCausalLMWithValueHead.from_pretrained(reward_model_path, num_labels=1)
    return model.to(device).eval()