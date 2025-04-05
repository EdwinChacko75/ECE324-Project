import torch
from transformers import GenerationConfig
from tqdm import tqdm
from torch.optim import Adam
from .modelling import compute_reward, compute_logprob


def compute_rlhf_loss(policy_model, rewards, input_ids, attention_mask, device, seq_len, compute_logprob, compute_reward):
    outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask)
    values = outputs[2].squeeze(-1)
    batch_indices = torch.arange(values.size(0), device=device)
    value_estimates = values[batch_indices, seq_len - 1]

    log_probs = compute_logprob(policy_model, input_ids, attention_mask)
    gen_log_probs = log_probs[batch_indices, seq_len - 1]

    advantage = rewards - value_estimates
    policy_loss = -gen_log_probs * advantage.detach()
    value_loss = (value_estimates - rewards) ** 2

    return policy_loss.mean() + 0.5 * value_loss.mean()

def generate_rollout(policy_model, input_ids, attention_mask, gen_config):
    with torch.no_grad():
        generated = policy_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_config.to_dict()
        )

    prompt_len = input_ids.shape[1]
    generated_tokens = generated[:, prompt_len:]

    combined_ids = torch.cat([input_ids, generated_tokens], dim=1)
    gen_attention = torch.ones_like(generated_tokens)
    combined_attention_mask = torch.cat([attention_mask, gen_attention], dim=1)

    return combined_ids, combined_attention_mask, combined_ids.shape[1]

def run_training_loop(config, policy_model, reward_model, dataloader, tokenizer, device):
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

            combined_ids, combined_attention_mask, seq_len = generate_rollout(
                policy_model, input_ids, attention_mask, gen_config
            )

            with torch.no_grad():
                rewards = compute_reward(reward_model, combined_ids, combined_attention_mask, seq_len)

            loss = compute_rlhf_loss(
                policy_model, rewards,
                combined_ids, combined_attention_mask,
                device, seq_len,
                compute_logprob, compute_reward
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_bar.set_postfix(loss=loss.item())

    print("RLHF policy training complete!")
    output_dir = config["training"]["rlhf"]["output_dir"]
    if config["training"]["rlhf"].get("use_lora", False):
        model = policy_model.merge_and_unload()
        model.save_pretrained(output_dir)
    else:
        policy_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)