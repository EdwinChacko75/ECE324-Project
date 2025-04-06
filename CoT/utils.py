import os
import json
import datetime
import yaml
import torch.distributed as dist

def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config= yaml.safe_load(f)
    config["learning_rate"] = float(config["learning_rate"])
    config["weight_decay"] = float(config.get("weight_decay", 0.0))
    config["warmup_steps"] = int(config.get("warmup_steps", 0))
    config["epochs"] = int(config["epochs"])
    config["batch_size"] = int(config["batch_size"])
    config["eval_batch_size"] = int(config.get("eval_batch_size", config["batch_size"]))
    config["gradient_accumulation_steps"] = int(config.get("gradient_accumulation_steps", 1))
    config["logging_steps"] = int(config.get("logging_steps", 50))
    config["eval_steps"] = int(config.get("eval_steps", 100))
    config["save_steps"] = int(config.get("save_steps", 200))
    config["save_total_limit"] = int(config.get("save_total_limit", 2))
    return config

def create_run_directory(base_dir="checkpoints", model_name="run", config=None):
    """
    Creates a unique logging directory for each run.
    TODO: Config logging
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
    output_file,
    batch_index,
    prompts,
    generated_texts,
    ground_truth_values,
    batch_acc,
    cumulative_accuracy,
):
    """
    Saves model outputs, corresponding prompts, and ground truth values to a structured text file.
    """
    # Set the file hearder. Can maybe include config details later.
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
