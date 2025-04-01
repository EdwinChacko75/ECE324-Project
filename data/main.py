import os
import torch
from tqdm import tqdm

from model import load_model
from dataset import load_dataset, extract_final_number, compute_accuracy
from utils import save_outputs_to_file, create_run_directory, load_config, save_outputs_to_json, save_jsonl_append

# Load config
config = load_config()

# Optional: Set CUDA devices from config
if config.get("cuda_visible_devices"):
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

MODEL_NAME = config["model_name"]
LoRA_MODEL = config["use_lora"]
LOAD_MODEL_PTH = config["lora_weights_path"]
BATCH_SIZE = config["batch_size"]
PRESCISION = getattr(torch, config["precision"])  # float16 -> torch.float16

CP_DIR = os.path.join(os.getcwd(), config["checkpoint_dir"])
# RUN_DIR = create_run_directory(CP_DIR, MODEL_NAME[:4])
# OUTPUT_FILE = os.path.join(RUN_DIR, config["output_file_name"])
OUTPUT_FILE = config["output_file"]
def main():
    # Load dataset
    print("Loading Dataset...")
    _, dataloader = load_dataset(dataset_name=config["dataset_name"], batch_size=BATCH_SIZE, split=config['split'])

    # Load model
    print("Loading Model...")
    model, tokenizer = load_model(MODEL_NAME, PRESCISION, lora=LoRA_MODEL, weights_pth=LOAD_MODEL_PTH)
    model.eval()
    print("Running Inference...")
    json_output_path = OUTPUT_FILE

    accuracies = []
    accuracy = 0
    all_results =[]
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

        # batch_predictions = [extract_final_number(text) for text in generated_texts]
        # batch_acc = compute_accuracy(batch_predictions, batch_ground_truth_values)
        # accuracies.append(batch_acc)
        # accuracy = (accuracy * batch_idx + batch_acc) / (batch_idx + 1)

        # print(f"Cumulative Accuracy: {accuracy * 100:.2f}%")

        # save_outputs_to_file(
        #     OUTPUT_FILE,
        #     batch_idx,
        #     prompts,
        #     generated_texts,
        #     batch_ground_truth_values,
        #     batch_acc,
        #     accuracy,
        # )
        # Save as JSON in current working directory or in RUN_DIR

    print(f"Saved structured JSON output to {json_output_path}")

    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Model Accuracy: {sum(accuracies) / len(accuracies) * 100:.2f}%")


if __name__ == "__main__":
    main()
