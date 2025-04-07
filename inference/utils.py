# utils.py
import os
import json
import yaml
import datetime
import torch.distributed as dist
from typing import Optional, Dict, Any, List, Union


def is_main_process() -> bool:
    """
    Determines if the current process is the main process in a distributed setup.

    Returns:
        bool: True if it is the main process or if not running distributed.
    """
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def get_final_dir(model: Optional[str], run: str, cp: str) -> str:
    """
    Constructs the final directory path for saving outputs or checkpoints.

    Args:
        model (Optional[str]): Path to the model checkpoint (can be None).
        run (str): Run directory name.
        cp (str): Base checkpoints directory.

    Returns:
        str: Path to the final directory.
    """
    if model is not None:
        model_identifier = (
            f"{os.path.basename(os.path.dirname(model))}{os.path.basename(run)[4:]}"
        )
        return os.path.join(cp, model_identifier)
    else:
        return run


def save_outputs_to_json(
    output_path: str, all_outputs: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> None:
    """
    Saves the outputs to a JSON file.

    Args:
        output_path (str): Path to the JSON output file.
        all_outputs (dict or list): Outputs to be serialized and saved.
    """
    with open(output_path, "w") as f:
        json.dump(all_outputs, f, indent=2)


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


def create_run_directory(
    base_dir: str = "checkpoints",
    model_name: str = "run",
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Creates a timestamped run directory and optionally saves the config.

    Args:
        base_dir (str): Root directory for saving.
        model_name (str): Name prefix for the run.
        config (dict, optional): Config to save in the run directory.

    Returns:
        str: Path to the created run directory.
    """
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%d_%H-%M")
    run_name = f"{model_name}_{timestamp}"
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    if config:
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

    return run_dir


def save_outputs_to_file(
    output_file: str,
    batch_index: int,
    prompts: List[str],
    generated_texts: List[str],
    ground_truth_values: List[str],
    batch_acc: float,
    cumulative_accuracy: float,
) -> None:
    """
    Appends model output logs to a text file in human-readable format.

    Args:
        output_file (str): Path to the output file.
        batch_index (int): Index of the current batch.
        prompts (list): List of input prompts.
        generated_texts (list): Corresponding model outputs.
        ground_truth_values (list): Ground truth answers.
        batch_acc (float): Accuracy of the current batch.
        cumulative_accuracy (float): Running average accuracy.
    """
    if batch_index == 0:
        with open(output_file, "w") as f:
            f.write("Model Outputs Log\n")
            f.write("=========================================\n")

    with open(output_file, "a") as f:
        f.write(f"\n=== Batch {batch_index} ===\n")
        f.write(f"Batch Accuracy: {batch_acc}\n")
        f.write(f"Cumulative Accuracy: {cumulative_accuracy}\n")

        for i, (prompt, output, ground_truth) in enumerate(
            zip(prompts, generated_texts, ground_truth_values)
        ):
            f.write(f"\nExample {i+1}:\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated Output: {output}\n")
            f.write(f"True Value: {ground_truth}\n")
            f.write("-----------------------------------------\n")
