import os
import torch
from torch.optim import AdamW

from model import load_model
from dataset import load_dataset
from utils import create_run_directory
from train import train_model, test_model

# Define the model and devices being used
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# os.environ['CUDA_VISIBLE_DEVICES'] = "3,4,5"
# MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# File paths
CP_DIR = os.path.join(os.getcwd(), "checkpoints/")
RUN_DIR = create_run_directory(CP_DIR, MODEL_NAME[:4])
OUTPUT_FILE = os.path.join(RUN_DIR, "outputs.txt")
# TODO: Set up a config file
CONFIG_FILE = ""

BATCH_SIZE = 32
PRESCISION = torch.float16  # Can adjust as needed torch.float32


def main():
    # Get the dataloader
    print("Loading Dataset...")
    _, test_loader = load_dataset(dataset_name="gsm8k", batch_size=BATCH_SIZE, split='test')
    _, train_loader = load_dataset(dataset_name="gsm8k", batch_size=BATCH_SIZE, split='train')

    # Get the model and tokenizer
    print("Loading Model...")
    model, tokenizer = load_model(MODEL_NAME, PRESCISION)


    epochs = 20
    learning_rate = 1e-4
    weight_decay = 0.01

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,                
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8                  
    )

    print("Training Model...")

    train_model(
        model,
        tokenizer,
        train_loader,
        PRESCISION,
        epochs,
        OUTPUT_FILE,
        optimizer,
        scheduler=None,
    )
    print("Testing Model...")

    test_model(model, tokenizer, test_loader, PRESCISION)

if __name__ == "__main__":
    main()
