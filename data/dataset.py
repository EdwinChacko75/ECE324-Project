import datasets
from torch.utils.data import DataLoader
import re
from fractions import Fraction
import math



BATCH_SIZE = 32

# for gsm8k
def prepare_example_1(example):
    """
    Prepare a single example for inference:
      - Extract the math problem as the question.
      - Extract the detailed ground truth chain-of-thought.
      - Parse out the final answer (ground truth value) which is preceded by "\boxed{".
      - Format the prompt for the LLM.
    """
    question = example.get("question")
    ground_truth = example.get("answer")
    
    ground_truth_value = None
    if ground_truth and "####" in ground_truth:
        ground_truth_value = ground_truth.split("####")[-1].strip()

    prompt = (
        f"Question: {question}\n"
        f"Final Answer: {ground_truth_value}\n"
        f"Original Steps:\n{ground_truth}\n\n"
        "Now, rewrite this explanation in a clear, step-by-step format using natural language.\n"
        "Start each step with a brief description of what the step is doing (e.g., 'Step N: [Describe goal of step].').\n"
        "Then explain the logic and math in that step.\n"
        "Always conclude with: \\boxed{<answer>}."
    )



    return {
        "prompt": prompt,
        "ground_truth": ground_truth,
        "ground_truth_value": ground_truth_value,
    }
    
# for math500
def prepare_example_2(example):
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
def prepare_example_3(example):
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
    


def collate_fn(batch):
    """
    Collate function for batching.
    """
    prompts = [item["prompt"] for item in batch]
    ground_truths = [item["ground_truth"] for item in batch]
    ground_truth_values = [item["ground_truth_value"] for item in batch]
    return {
        "prompts": prompts,
        "ground_truths": ground_truths,
        "ground_truth_values": ground_truth_values,
    }


def extract_final_number(text):
    """
    Extracts the final boxed answer from the model's output.
    - Supports spaces inside \boxed{}
    - Handles integers, decimals, negative numbers, and scientific notation.
    """
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
    """
    Computes accuracy by comparing lists of predicted and ground truth numbers.
    """
    # correct = 0
    # for pred, truth in zip(predicted_values, ground_truth_values):
    #     if isinstance(pred, str):
    #         pred = extract_final_number(pred)
    #     if isinstance(pred, str):
    #         continue
    #     if isinstance(pred, (int, float)) and truth is not None and math.isclose(pred, float(truth.replace(",", "")), rel_tol=1e-6):
    #         correct +=1

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
    """
    Converts Math500-style answers into numerical or text format.
    Ensures ground truth values are evaluated correctly.
    """
    if not answer:
        return None

    answer = str(answer).strip()

    # Handle text-based answers (e.g., \text{Evelyn})
    text_match = re.match(r"\\text\{(.+?)\}", answer)
    if text_match:
        return text_match.group(1).strip()  # Extract text

    # Handle mathematical expressions
    try:
        return latex2sympy(answer).evalf()  # Convert to numeric value
    except:
        return answer  # Fallback for unknown cases
    
def general_compute_accuracy(predictions, ground_truths):
    """
    Computes accuracy by comparing predictions and ground truths.
    Handles both numerical and non-numerical (textual) answers.
    """
    correct = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        pred_processed = process_latex_answer(pred)
        gt_processed = process_latex_answer(gt)
        
        # Normalize predictions and ground truth (strip whitespaces, lowercase)
        pred_normalized = str(pred_processed).strip().lower()
        gt_normalized = str(gt_processed).strip().lower()

        # # Skip if either is None (invalid input)
        # if pred_normalized is None or gt_normalized is None:
        #     continue

        # # Check if both are numbers and use approximate comparison
        # if isinstance(pred_normalized, float) and isinstance(gt_normalized, float):
        #     if math.isclose(pred_normalized, gt_normalized, rel_tol=1e-6):
        #         correct += 1
        #         continue

        # Fallback to exact string match for text
        if pred_normalized == gt_normalized:
            correct += 1

    return correct / total if total > 0 else 0

def load_dataset(dataset_name="gsm8k", batch_size=8, split=None, collate_fn=collate_fn):
    """
    Get the dataloader for a specified dataset.
    """
    # Load dataset and prepare it
    
    # testing data
    if dataset_name == "gsm8k":
        dataset = datasets.load_dataset(dataset_name, "main", split=split)
        prepared_dataset = dataset.map(prepare_example_1, load_from_cache_file=False)
    elif dataset_name == "HuggingFaceH4/MATH-500":
        dataset = datasets.load_dataset(dataset_name, split=split)
        prepared_dataset = dataset.map(prepare_example_2, load_from_cache_file=False)
        
    # training data
    elif dataset_name == "Maxwell-Jia/AIME_2024":
        dataset = datasets.load_dataset(dataset_name, split=split)  
        prepared_dataset = dataset.map(prepare_example_3, load_from_cache_file=False)
    
    
    ### can add more datasets
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    
    # breakpoint()
    # prepared_dataset = dataset.map(prepare_example, load_from_cache_file=False)
    # prepared_dataset =dataset
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
