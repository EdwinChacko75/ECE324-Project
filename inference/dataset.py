import datasets
from torch.utils.data import DataLoader
import re
from fractions import Fraction
import math


# for gsm8k
def gsm8k_instruct(example):
    question = example.get("question")
    ground_truth = example.get("answer")
    
    prompt = (
        "Question: If you buy 3 apples for $2 each and a banana for $1, how much do you spend?\n"
        "Answer: Let's solve it step by step.\n"
        "Step 1: Calculate the cost of the apples. 3 * $2 = $6.\n"
        "Step 2: The banana costs $1.\n"
        "Step 3: Add the costs together. $6 + $1 = \\boxed{7}\n"
        "Total cost = \\boxed{7}\n"
        f"Question: {question}\n"
        "Answer: Let's solve it step by step.\n"
    )

    ground_truth_value = None
    if ground_truth and "####" in ground_truth:
        ground_truth_value = ground_truth.split("####")[-1].strip()

    return {
        "prompt": prompt,
        "ground_truth": ground_truth,
        "ground_truth_value": ground_truth_value,
    }
    
# for math500
def math500_instruct(example):
    question = example.get("problem")
    ground_truth = example.get("solution")

    p = (
        f"Question: {question}\n"
    )

    ground_truth_value = example.get("answer")
    

    return {
        "prompt": p,
        "ground_truth": ground_truth,
        "ground_truth_value": ground_truth_value,
    }

# for aime2024
def aime2024_instruct(example):
    question = example.get("Problem")
    ground_truth = example.get("Solution")

    p = (
        f"Question: {question}\n"
    )

    ground_truth_value = str(example.get("Answer"))
    

    return {
        "prompt": p,
        "ground_truth": ground_truth,
        "ground_truth_value": ground_truth_value,
    }
    
# for logiqa
def logiqa_instruct(example):
    context = example.get("context")
    query = example.get("query")
    options = example.get("options")
    
    question = context + ' ' + query + ' ' + str(options)
    ground_truth = None
    ground_truth_value = str(example.get("Answer"))

    p = (
        f"Question: {question}\n"
    )

    return {
        "prompt": p,
        "ground_truth": ground_truth,
        "ground_truth_value": ground_truth_value,
    }
    
def collate_fn(batch):
    prompts = [item["prompt"] for item in batch]
    ground_truths = [item["ground_truth"] for item in batch]
    ground_truth_values = [item["ground_truth_value"] for item in batch]
    return {
        "prompts": prompts,
        "ground_truths": ground_truths,
        "ground_truth_values": ground_truth_values,
    }


def extract_final_number(text):
    if not text:
        return None

    match = re.findall(r"\\boxed{(.*?)}", text)

    if not match or match == []:
        num_match = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?|[-+]?\d+/\d+", text)
        if not num_match:
            return None
        return float(num_match[-1]) if "." in num_match[-1] else int(num_match[-1])

    num_match = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?|[-+]?\d+/\d+", text)
    num_match = None if not num_match else num_match[-1]

    num_str = match[-1].replace(",", "").strip()

    try:
        if "." in num_str or "e" in num_str:
            return float(num_str)
        if "/" in num_str:
            return float(Fraction(num_str))
        return int(num_str)
    except:
        return float(num_match)


def compute_accuracy(predicted_values, ground_truth_values):
    correct = sum(
        1
        for pred, truth in zip(predicted_values, ground_truth_values)
        if isinstance(pred, (int, float))
        and truth is not None
        and math.isclose(pred, float(truth.replace(",", "")), rel_tol=1e-6)
    )

    total = len(ground_truth_values)
    return correct / total if total > 0 else 0


def process_latex_answer(answer):
    if not answer:
        return None

    answer = str(answer).strip()

    # Handle text-based answers (e.g., \text{Evelyn})
    text_match = re.match(r"\\text\{(.+?)\}", answer)
    if text_match:
        return text_match.group(1).strip()  # Extract text
    return answer  # Fallback for unknown cases
    
def general_compute_accuracy(predictions, ground_truths):
    correct = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        pred_processed = process_latex_answer(pred)
        gt_processed = process_latex_answer(gt)
        
        # Normalize predictions and ground truth (strip whitespaces, lowercase)
        pred_normalized = str(pred_processed).strip().lower()
        gt_normalized = str(gt_processed).strip().lower()

        if pred_normalized == gt_normalized:
            correct += 1

    return correct / total if total > 0 else 0

def load_dataset(dataset_name="gsm8k", batch_size=8, collate_fn=collate_fn, instruct=None):
    # Load dataset and prepare it
    
    # testing data
    if dataset_name == "gsm8k":
        dataset = datasets.load_dataset(dataset_name, "main", split="test")
        prepared_dataset = dataset.map(gsm8k_instruct, load_from_cache_file=False)
    elif dataset_name == "HuggingFaceH4/MATH-500":
        dataset = datasets.load_dataset(dataset_name, split="test")
        prepared_dataset = dataset.map(math500_instruct, load_from_cache_file=False)
    elif dataset_name == "lucasmccabe/logiqa/test":
        dataset = datasets.load_dataset("lucasmccabe/logiqa", split="test")
        prepared_dataset = dataset.map(logiqa_instruct, load_from_cache_file=False)
        
    # training data
    elif dataset_name == "Maxwell-Jia/AIME_2024":
        dataset = datasets.load_dataset(dataset_name, split="train")  
        prepared_dataset = dataset.map(aime2024_instruct, load_from_cache_file=False)
    elif dataset_name == "lucasmccabe/logiqa/train":
        dataset = datasets.load_dataset("lucasmccabe/logiqa", split="train")
        prepared_dataset = dataset.map(logiqa_instruct, load_from_cache_file=False)
    
    # validation data
    elif dataset_name == "lucasmccabe/logiqa/validation":
        dataset = datasets.load_dataset("lucasmccabe/logiqa", split="validation")
        prepared_dataset = dataset.map(logiqa_instruct, load_from_cache_file=False)
    ### can add more datasets
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    prepared_dataset = prepared_dataset.map(
        lambda x: {"prompt_length": len(x["prompt"])},
	load_from_cache_file=False
    )

    # sort in reverse to get upper bound
    prepared_dataset = prepared_dataset.sort("prompt_length", reverse=True)

    dataloader = DataLoader(
        prepared_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dataset, dataloader

