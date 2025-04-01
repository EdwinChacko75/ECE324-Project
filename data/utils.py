import os
import json
import datetime
import yaml

def save_jsonl_append(file_path, examples):
    """
    Appends a list of dicts (examples) to a JSONL file.
    """
    with open(file_path, "a") as f:
        for ex in examples:
            json.dump(ex, f)
            f.write("\n")
            
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
