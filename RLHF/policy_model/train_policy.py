
from .model import load_tokenizer, load_policy_model, load_reward_model
from .data import load_rlhf_datasets
from .train import run_training_loop
import torch

def train_rlhf_policy(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = load_tokenizer(config["model"]["base_model"])
    dataloader = load_rlhf_datasets(config, tokenizer)
    policy_model = load_policy_model(config, device)
    reward_model = load_reward_model(config, device)

    run_training_loop(config, policy_model, reward_model, dataloader, tokenizer, device)
