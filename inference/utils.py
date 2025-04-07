import os
import json
import datetime
import yaml
import torch.distributed as dist
 
def is_main_process():
     return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def get_final_dir(model, run, cp):
    if model is not None:
        model_identifier = f"{os.path.basename(os.path.dirname(model))}{os.path.basename(run)[4:]}"
        return os.path.join(cp, model_identifier)
    else:
        return run

def save_outputs_to_json(output_path, all_outputs):
    with open(output_path, "w") as f:
        json.dump(all_outputs, f, indent=2)

def save_jsonl_append(file_path, examples):
    with open(file_path, "a") as f:
        for ex in examples:
            json.dump(ex, f)
            f.write("\n")

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def create_run_directory(base_dir="checkpoints", model_name="run", config=None):
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


def save_outputs_to_file(output_file,batch_index,prompts,generated_texts,ground_truth_values,batch_acc,
    cumulative_accuracy):
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