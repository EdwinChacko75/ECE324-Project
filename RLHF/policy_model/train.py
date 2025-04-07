# train.py
import os
import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import GenerationConfig, PreTrainedTokenizerBase
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any


def init_distributed() -> Optional[int]:
    """
    Initializes PyTorch DistributedDataParallel (DDP) if environment variables are set.

    Returns:
        int or None: The local rank of the process if DDP is initialized, otherwise None.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(
            f"Distributed training: local_rank {local_rank}, world_size {dist.get_world_size()}"
        )
        return local_rank
    return None


def compute_reward(
    model: Module, input_ids: Tensor, attention_mask: Tensor, seq_len: int
) -> Tensor:
    """
    Computes scalar rewards using the value head of the reward model.

    Args:
        model: Reward model with a value head.
        input_ids: Tokenized input sequences.
        attention_mask: Attention masks.
        seq_len: Sequence length (to extract final token reward).

    Returns:
        Tensor: A tensor of scalar rewards.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    values = outputs[2].squeeze(-1)
    batch_indices = torch.arange(values.size(0), device=values.device)
    return values[batch_indices, seq_len - 1]


def compute_logprob(model: Module, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Computes token-level log-probabilities from the model output.

    Args:
        model: Policy model.
        input_ids: Tokenized input sequences.
        attention_mask: Attention masks.

    Returns:
        Tensor: Log-probabilities of each token.
    """
    logits, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def compute_rlhf_loss(
    policy_model: Module,
    rewards: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    device: torch.device,
    seq_len: int,
    compute_logprob_fn,
    compute_reward_fn,
) -> Tensor:
    """
    Computes the RLHF loss (policy + value loss) for one batch.

    Args:
        policy_model: Model being fine-tuned.
        rewards: Target rewards for generated sequences.
        input_ids: Tokenized inputs (prompts + responses).
        attention_mask: Corresponding attention mask.
        device: Device for computation.
        seq_len: Length of the generated response.
        compute_logprob_fn: Function to compute log-probs.
        compute_reward_fn: Function to compute rewards.

    Returns:
        Tensor: Scalar loss value for backpropagation.
    """
    outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask)
    values = outputs[2].squeeze(-1)
    batch_indices = torch.arange(values.size(0), device=device)
    value_estimates = values[batch_indices, seq_len - 1]

    log_probs = compute_logprob_fn(policy_model, input_ids, attention_mask)
    gen_log_probs = log_probs[batch_indices, seq_len - 1]

    advantage = rewards - value_estimates
    policy_loss = -gen_log_probs * advantage.detach()
    value_loss = (value_estimates - rewards) ** 2

    return policy_loss.mean() + 0.5 * value_loss.mean()


def generate_rollout(
    policy_model: Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    gen_config: GenerationConfig,
) -> Tuple[Tensor, Tensor, int]:
    """
    Generates responses from the policy model based on prompts.

    Args:
        policy_model: Policy model (can be DDP or not).
        input_ids: Tokenized prompts.
        attention_mask: Attention mask for the prompts.
        gen_config: Generation config (sampling strategy, max tokens, etc.).

    Returns:
        Tuple:
            - combined_ids: Prompt + response token IDs.
            - combined_attention_mask: Corresponding attention mask.
            - seq_len: Total length of the sequence (prompt + response).
    """
    model = policy_model.module if hasattr(policy_model, "module") else policy_model

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **gen_config.to_dict()
        )

    prompt_len = input_ids.shape[1]
    generated_tokens = generated[:, prompt_len:]
    combined_ids = torch.cat([input_ids, generated_tokens], dim=1)
    gen_attention = torch.ones_like(generated_tokens)
    combined_attention_mask = torch.cat([attention_mask, gen_attention], dim=1)

    return combined_ids, combined_attention_mask, combined_ids.shape[1]


def run_training_loop(
    config: Dict[str, Any],
    policy_model: Module,
    reward_model: Module,
    dataloader: DataLoader,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> None:
    """
    Runs the full RLHF training loop with PPO-style reward feedback.

    Args:
        config: Full training configuration.
        policy_model: Model to be trained with PPO loss.
        reward_model: Static reward model with value head.
        dataloader: Dataloader yielding tokenized prompt batches.
        tokenizer: Tokenizer used for saving model later.
        device: CUDA device or CPU.
    """
    optimizer = Adam(
        policy_model.parameters(), lr=config["training"]["rlhf"]["learning_rate"]
    )

    gen_config = GenerationConfig(
        max_new_tokens=config["training"]["rlhf"].get("max_new_tokens", 50),
        do_sample=True,
        temperature=1.0,
    )

    num_epochs = config["training"]["rlhf"]["epochs"]

    for epoch in range(num_epochs):
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in epoch_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            combined_ids, combined_attention_mask, seq_len = generate_rollout(
                policy_model, input_ids, attention_mask, gen_config
            )

            with torch.no_grad():
                rewards = compute_reward(
                    reward_model, combined_ids, combined_attention_mask, seq_len
                )

            loss = compute_rlhf_loss(
                policy_model,
                rewards,
                combined_ids,
                combined_attention_mask,
                device,
                seq_len,
                compute_logprob,
                compute_reward,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_bar.set_postfix(loss=loss.item())

        if dist.is_initialized():
            dist.barrier()

    print("RLHF policy training complete!")

    # Save final model and tokenizer (only on main process)
    if not dist.is_initialized() or dist.get_rank() == 0:
        output_dir = config["training"]["rlhf"]["output_dir"]
        print(f"Saving model to {output_dir}")

        if config["training"]["rlhf"].get("lora", False):
            print("Merging LoRA adapters before saving.")
            lora_merged = (
                policy_model.module.merge_and_unload()
                if dist.is_initialized()
                else policy_model.merge_and_unload()
            )
            model_to_save = (
                lora_merged.base_model.model
                if hasattr(lora_merged, "base_model")
                else lora_merged
            )
        else:
            model_to_save = (
                policy_model.module.base_model.model
                if dist.is_initialized()
                else policy_model.base_model.model
            )

        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
