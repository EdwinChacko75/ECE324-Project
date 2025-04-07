import yaml
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file and ensures specific nested learning rate fields are cast to float.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Parsed and type-corrected configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Ensure learning rates are floats
    config["training"]["reward_model"]["learning_rate"] = float(
        config["training"]["reward_model"]["learning_rate"]
    )
    config["training"]["rlhf"]["learning_rate"] = float(
        config["training"]["rlhf"]["learning_rate"]
    )

    return config
