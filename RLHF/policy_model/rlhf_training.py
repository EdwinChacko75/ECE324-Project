import torch
from torch.optim import Adam
from transformers import AutoTokenizer, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from tqdm import tqdm
from peft import LoraConfig, get_peft_model


def load_tokenizer(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_rlhf_datasets(config, tokenizer):
    dataset_path = config["dataset"]["train_path"]
    dataset = load_dataset("json", data_files={"train": dataset_path})["train"]

    def tokenize_fn(examples):
        return tokenizer(examples["input"], truncation=True, max_length=config["model"]["max_length"], padding="max_length")

    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["training"]["rlhf"]["batch_size"], shuffle=True)
    return dataloader


def compute_reward(model, input_ids, attention_mask, seq_len):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    values = outputs[2].squeeze(-1)
    batch_indices = torch.arange(values.size(0), device=values.device)
    reward = values[batch_indices, seq_len - 1]
    return reward


def compute_logprob(model, input_ids, attention_mask):
    logits, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    selected_logprobs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    return selected_logprobs


def train_rlhf_policy(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_model = config["model"]["base_model"]
    reward_model_path = config["training"]["rlhf"]["reward_model_dir"]
    output_dir = config["training"]["rlhf"]["output_dir"]

    # Load tokenizer and datasets
    tokenizer = load_tokenizer(base_model)
    dataloader = load_rlhf_datasets(config, tokenizer)

    # Load policy model and optionally apply LoRA
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model, num_labels=1)
    if config["training"]["rlhf"].get("use_lora", False):
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(policy_model, lora_config)
        # policy_model.print_trainable_parameters()
    policy_model.to(device)

    # Load frozen reward model
    reward_model = AutoModelForCausalLMWithValueHead.from_pretrained(reward_model_path, num_labels=1).to(device)
    reward_model.eval()

    # Set up optimizer and generation config
    optimizer = Adam(policy_model.parameters(), lr=config["training"]["rlhf"]["learning_rate"])
    gen_config = GenerationConfig(
        max_new_tokens=config["training"]["rlhf"].get("max_new_tokens", 50),
        do_sample=True,
        temperature=1.0,
    )

    num_epochs = config["training"]["rlhf"]["epochs"]

    for epoch in range(num_epochs):
        epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in epoch_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.no_grad():
                generated = policy_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_config.to_dict()
                )

            prompt_length = input_ids.shape[1]
            generated_tokens = generated[:, prompt_length:]
            combined_ids = torch.cat([input_ids, generated_tokens], dim=1)
            gen_attention = torch.ones_like(generated_tokens)
            combined_attention_mask = torch.cat([attention_mask, gen_attention], dim=1)

            combined_seq_len = combined_ids.shape[1]
            with torch.no_grad():
                rewards = compute_reward(reward_model, combined_ids, combined_attention_mask, combined_seq_len)

            outputs = policy_model(input_ids=combined_ids, attention_mask=combined_attention_mask)
            values = outputs[2].squeeze(-1)
            batch_indices = torch.arange(values.size(0), device=device)
            value_estimates = values[batch_indices, combined_seq_len - 1]

            log_probs = compute_logprob(policy_model, combined_ids, combined_attention_mask)
            gen_log_probs = log_probs[batch_indices, combined_seq_len - 1]

            advantage = rewards - value_estimates
            policy_loss = -gen_log_probs * advantage.detach()
            value_loss = (value_estimates - rewards) ** 2
            loss = policy_loss.mean() + 0.5 * value_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_bar.set_postfix(loss=loss.item())

    print("RLHF policy training complete!")
    if config["training"]["rlhf"].get("lora", False):
        model = model.merge_and_unload()
        policy_model.save_pretrained(output_dir)
    else:
        policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
