import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(MODEL_NAME, PRESCISION):
    """
    Load the specified model with the specified prescision.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # llama models dont have pad_token
    # TODO: this is erroneous for models where pad already exists
    tokenizer.pad_token = tokenizer.eos_token 

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=PRESCISION,
        device_map="auto"  
    )

    return model, tokenizer

def run_inference():
    """
    Perform inference with all GPUs collaborating.
    Used for testing.
    """
    model, tokenizer = load_model()

    input_text = "The quick brown fox"
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")  

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float32):  
            outputs = model.generate(**inputs, max_length=1000)

    print("Generated Output: ", tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    run_inference()
