# rlhf_training.py
import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from data_loader import load_prm800k


# TODO Nothing here works


def reward_function(model, tokenizer, texts):
    """
    Computes rewards for generated texts using the provided model.
    This dummy function uses the third logit's softmax score as the reward.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        rewards = (probs[:, 2] - probs[:, 0]).tolist()

        rewards = probs[:, 2].tolist()  # Use class index 2 as the reward signal
    return rewards

def train_rlhf_policy(config):
    # Load the saved checkpoint from reward model training
    saved_path = config["training"]["reward_model"]["output_dir"]
    saved_path = config["model"]["base_model"]

    # Load tokenizer from the saved checkpoint
    tokenizer = AutoTokenizer.from_pretrained(saved_path)
    
    # Load the policy model with a value head for RL training
    policy_model = AutoModelForCausalLMWithValueHead.from_pretrained(saved_path)
    # Create a reference model (frozen copy) of the policy model
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(saved_path)
    ref_model.eval()

    # Load the RLHF dataset (assumed to be in JSONL format with an "input" field)
    dataset = load_prm800k(config["dataset"]["path"], split=config["dataset"]["split"])
    
    # Tokenize the dataset
    def tokenize_fn(example):
        return tokenizer(example["input"], truncation=True, max_length=config["model"]["max_length"])
    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Initialize PPO configuration 
    ppo_config = PPOConfig(
        learning_rate=config["training"]["rlhf"]["learning_rate"],
        batch_size=config["training"]["rlhf"]["batch_size"],
        ppo_epochs=config["training"]["rlhf"]["epochs"],
        log_with=None,
    )
    
    # Create the PPO trainer 
    ppo_trainer = PPOTrainer(
        policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=lambda data: {
            "input_ids": torch.stack([d["input_ids"] for d in data]),
            "attention_mask": torch.stack([d["attention_mask"] for d in data])
        },
        **ppo_config.__dict__,
    )
    
    # Move models to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_model.to(device)
    ref_model.to(device)
    
    # RLHF training loop using PPO
    for epoch in range(config["training"]["rlhf"]["epochs"]):
        print(f"Starting epoch {epoch+1}/{config['training']['rlhf']['epochs']}")
        for batch in ppo_trainer.dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Generate responses using the policy model
            response_ids = ppo_trainer.generate(input_ids, max_new_tokens=config["training"]["rlhf"]["max_new_tokens"])
            responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            
            # Compute rewards using the reward function
            rewards = reward_function(policy_model, tokenizer, responses)
            
            # Perform a PPO step to update the policy model
            stats = ppo_trainer.step(input_ids, response_ids, rewards)
            print(f"Epoch {epoch+1} step stats: {stats}")
    
    # Save the updated policy model to the specified output directory
    policy_model.save_pretrained(config["training"]["rlhf"]["output_dir"])
    tokenizer.save_pretrained(config["training"]["rlhf"]["output_dir"])

if __name__ == "__main__":
    # can integrate run it directly.
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_rlhf_policy(config)
