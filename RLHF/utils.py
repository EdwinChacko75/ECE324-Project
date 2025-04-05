# utils.py
import yaml

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["training"]["reward_model"]["learning_rate"] = float(config["training"]["reward_model"]["learning_rate"])
    config["training"]["rlhf"]["learning_rate"] = float(config["training"]["rlhf"]["learning_rate"])
    return config
