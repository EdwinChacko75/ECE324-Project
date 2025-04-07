import os
import torch
from tqdm import tqdm

from model import load_model
from dataset import load_dataset, extract_final_number
from utils import load_config, save_jsonl_append
from clean_data import process_inference_file, validate_data

# Load config
config = load_config()

# Optional: Set CUDA devices from config
if config.get("cuda_visible_devices"):
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

DATASET = config["dataset_name"]
MODEL_NAME = config["model_name"]
BATCH_SIZE = config["batch_size"]
PRESCISION = getattr(torch, config["precision"])  # float16 -> torch.float16

CP_DIR = os.path.join(os.getcwd(), config["checkpoint_dir"])
OUTPUT_FILE = config["output_file"]
def main():
    # Load dataset
    print("Loading Dataset...")
    _, dataloader = load_dataset(dataset_name=DATASET, batch_size=BATCH_SIZE, split=config['split'])

    # Load model
    print("Loading Model...")
    model, tokenizer = load_model(MODEL_NAME, PRESCISION, lora=False, weights_pth=None)
    model.eval()
    print("Running Inference...")
    json_output_path = OUTPUT_FILE

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing Batches")):
        prompts = batch["prompts"]
        batch_ground_truth_values = batch["ground_truth_values"]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

        with torch.no_grad():
            with torch.autocast("cuda", dtype=PRESCISION):
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=config["max_new_tokens"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    do_sample=config["do_sample"],
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    early_stopping=config.get("early_stopping", False),
                    repetition_penalty=config.get("repetition_penalty", 1.0),
                    num_beams=config.get("num_beams", 1.0),
                )

        generated_texts = [
            tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]
        batch_results = [
            {
                "prompt": prompts[i],
                "generated_solution": generated_texts[i],
                "ground_truth_value": batch_ground_truth_values[i],
                "predicted_value": extract_final_number(generated_texts[i]),
            }
            for i in range(len(prompts))
        ]
        save_jsonl_append(json_output_path, batch_results)

    print(f"Saved structured JSON output to {json_output_path}")

    # Clean the data
    output_jsonl = os.path.join(os.path.dirname(json_output_path), f"{DATASET}_{config['split']}.jsonl")
    process_inference_file(json_output_path, output_jsonl)
    validate_data(output_jsonl)

if __name__ == "__main__":
    main()
