import os
import torch
import torch.distributed as dist
from .model import load_tokenizer, load_policy_model, load_reward_model
from .data import load_rlhf_datasets
from .train import run_training_loop

def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"Distributed training: local_rank {local_rank}, world_size {dist.get_world_size()}")
        return local_rank
    return None

def train_rlhf_policy(config):
    local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if local_rank is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = load_tokenizer(config["model"]["base_model"])
    dataloader = load_rlhf_datasets(config, tokenizer)
    policy_model = load_policy_model(config, device)
    reward_model = load_reward_model(config, device)

    run_training_loop(config, policy_model, reward_model, dataloader, tokenizer, device)
