import os
import torch
from tqdm import tqdm

from model import load_model
from dataset import load_dataset, extract_final_number, compute_accuracy
from utils import save_outputs_to_file, create_run_directory


# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4" 
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2" 
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# os.environ['CUDA_VISIBLE_DEVICES'] = "3,4,5" 
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  


BATCH_SIZE = 32
CONFIG_FILE = '' # set up later
CP_DIR = os.path.join(os.getcwd(), "checkpoints/")
RUN_DIR = create_run_directory(CP_DIR, MODEL_NAME[:4])
OUTPUT_FILE = os.path.join(RUN_DIR,'outputs.txt')


# PRESCISION = torch.float32
PRESCISION = torch.float16

def main():
    print("Loading Dataset...")
    _, dataloader = load_dataset(dataset_name="gsm8k", batch_size=BATCH_SIZE)

    print("Loading Model...")
    model, tokenizer = load_model(MODEL_NAME, PRESCISION)
    model.eval()
    print("Running Inference...")

    accuracies = []
    accuracy = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing Batches")):
        prompts = batch["prompts"]
        batch_ground_truth_values = batch["ground_truth_values"]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        # print(f"Input token length: {inputs["input_ids"].shape[1]}")

        with torch.no_grad():
            with torch.autocast("cuda", dtype=PRESCISION):
                outputs = model.generate(**inputs, max_length=1000)
        
        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        batch_predictions = [extract_final_number(text) for text in generated_texts]

        batch_acc = compute_accuracy(batch_predictions, batch_ground_truth_values)
        accuracies.append(batch_acc)
        accuracy = (accuracy * batch_idx + batch_acc) / ( (batch_idx + 1))
        print(f"Cumulative Accuracy: {accuracy * 100:.2f}%")

        # Save outputs to file
        save_outputs_to_file(OUTPUT_FILE, batch_idx, prompts, generated_texts, batch_ground_truth_values, batch_acc, accuracy)

    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(f"Model Accuracy: {sum(accuracies)/len(accuracies) * 100:.2f}%")
    assert(accuracy == sum(accuracies)/len(accuracies))

if __name__ == "__main__":
    main()