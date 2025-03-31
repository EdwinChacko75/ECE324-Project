import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


def load_model(MODEL_NAME, PRESCISION, lora=False, weights_pth=None):
    """
    Load the specified model with the specified prescision.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=PRESCISION, device_map="auto"
    )

    # TODO base model weights (non lora)
    # if weights_pth is not None and not lora:
    #     base_model.load_wegits()

    if not lora:
        print("Using the base model without LoRA.")
        return base_model, tokenizer

    # Configure LoRA parameters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    if weights_pth is not None:
        model = PeftModel.from_pretrained(base_model, weights_pth)
        print(f"Loaded LoRA weights from {weights_pth}")
    else:
        model = get_peft_model(model, lora_config)


    print("LoRA is enabled and has been applied to the model.")
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
