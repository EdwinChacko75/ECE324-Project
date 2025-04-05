import torch
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

def compute_reward(model, input_ids, attention_mask, seq_len):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    values = outputs[2].squeeze(-1)
    batch_indices = torch.arange(values.size(0), device=values.device)
    return values[batch_indices, seq_len - 1]

def compute_logprob(model, input_ids, attention_mask):
    logits, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
