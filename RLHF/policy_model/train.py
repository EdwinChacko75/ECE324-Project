import torch
from transformers import GenerationConfig
from tqdm import tqdm
from torch.optim import Adam
import torch.distributed as dist

def compute_reward(model, input_ids, attention_mask, seq_len):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    values = outputs[2].squeeze(-1)
    batch_indices = torch.arange(values.size(0), device=values.device)
    return values[batch_indices, seq_len - 1]

def compute_logprob(model, input_ids, attention_mask):
    logits, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

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
    
    # Safely get the underlying model if wrapped in DDP
    model = policy_model.module if hasattr(policy_model, "module") else policy_model

    with torch.no_grad():
        generated = model.generate(
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
        # If using a DistributedSampler, set the epoch to reshuffle data differently every epoch
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
            
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

        if dist.is_initialized():
            dist.barrier()
        

    print("RLHF policy training complete!")
    # Save the model only from the main process (rank 0)
    if not dist.is_initialized() or dist.get_rank() == 0:
        output_dir = config["training"]["rlhf"]["output_dir"]
        if config["training"]["rlhf"].get("lora", False):
            print(f"Saving model to {output_dir} with merged LoRA adapters.")

            # When using DDP, access the underlying module
            lora_merged = policy_model.module.merge_and_unload() if dist.is_initialized() else policy_model.merge_and_unload()
            model_to_save = lora_merged.base_model.model if hasattr(lora_merged, "base_model") else lora_merged
            model_to_save.save_pretrained(output_dir)
        else:
            print(f"Saving model to {output_dir}.")
            if dist.is_initialized():
                # model with value head inside DDP
                model_to_save = policy_model.module.base_model.model
            else:
                # model with value head without DDP
                model_to_save = policy_model.base_model.model
            model_to_save.save_pretrained(output_dir)

        tokenizer.save_pretrained(output_dir)
