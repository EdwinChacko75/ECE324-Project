# main.py
import argparse
from reward_model.reward_model import train_reward_model
from policy_model.train_policy import train_rlhf_policy
from utils import load_config


def main():
    parser = argparse.ArgumentParser(description="RLHF Training Pipeline")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["reward", "rlhf"],
        required=True,
        help="Task to run: reward (train reward model) or rlhf (train RLHF policy)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.task == "reward":
        train_reward_model(config)
    elif args.task == "rlhf":
        train_rlhf_policy(config)


if __name__ == "__main__":
    main()
