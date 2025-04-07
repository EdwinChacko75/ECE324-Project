# utils.py
import json
import yaml
from typing import Dict, Any, List


def save_jsonl_append(file_path: str, examples: List[Dict[str, Any]]) -> None:
    """
    Appends JSON lines (one object per line) to a .jsonl file.

    Args:
        file_path (str): File path to append to.
        examples (list): List of dictionaries to append.
    """
    with open(file_path, "a") as f:
        for ex in examples:
            json.dump(ex, f)
            f.write("\n")


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads a YAML config file.

    Args:
        config_path (str): Path to the YAML file.

    Returns:
        dict: Parsed configuration.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
