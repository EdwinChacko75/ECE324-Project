import re
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fractions import Fraction
import datasets
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader


# ============================
# Instruction Formatters
# ============================


def gsm8k_instruct(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Formats a GSM8K example into a prompt and extracts ground truth.

    Args:
        example (dict): A dictionary with 'question' and 'answer' keys.

    Returns:
        dict: Formatted prompt and extracted ground truth.
    """
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


def math500_instruct(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Formats a MATH-500 example into a prompt and extracts ground truth.

    Args:
        example (dict): A dictionary with 'problem', 'solution', and 'answer'.

    Returns:
        dict: Formatted prompt and extracted ground truth.
    """
    question = example.get("problem")
    ground_truth = example.get("solution")
    ground_truth_value = example.get("answer")

    prompt = f"Question: {question}\n"

    return {
        "prompt": prompt,
        "ground_truth": ground_truth,
        "ground_truth_value": ground_truth_value,
    }


def aime2024_instruct(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Formats an AIME 2024 example into a prompt and extracts ground truth.

    Args:
        example (dict): A dictionary with 'Problem', 'Solution', and 'Answer'.

    Returns:
        dict: Formatted prompt and extracted ground truth.
    """
    question = example.get("Problem")
    ground_truth = example.get("Solution")
    ground_truth_value = str(example.get("Answer"))

    prompt = f"Question: {question}\n"

    return {
        "prompt": prompt,
        "ground_truth": ground_truth,
        "ground_truth_value": ground_truth_value,
    }


def logiqa_instruct(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Formats a LogiQA example into a prompt and extracts ground truth.

    Args:
        example (dict): A dictionary with 'context', 'query', 'options', and 'Answer'.

    Returns:
        dict: Formatted prompt and extracted ground truth.
    """
    context = example.get("context")
    query = example.get("query")
    options = example.get("options")

    question = f"{context} {query} {str(options)}"
    ground_truth_value = str(example.get("Answer"))

    prompt = f"Question: {question}\n"

    return {
        "prompt": prompt,
        "ground_truth": None,
        "ground_truth_value": ground_truth_value,
    }


# ============================
# Collation
# ============================


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Collates a batch of dataset examples for inference.

    Args:
        batch (list): A list of examples with prompt and ground truth info.

    Returns:
        dict: Collated batch with separate fields.
    """
    prompts = [item["prompt"] for item in batch]
    ground_truths = [item["ground_truth"] for item in batch]
    ground_truth_values = [item["ground_truth_value"] for item in batch]
    return {
        "prompts": prompts,
        "ground_truths": ground_truths,
        "ground_truth_values": ground_truth_values,
    }


# ============================
# Evaluation Utilities
# ============================


def extract_final_number(text: str) -> Optional[Union[int, float]]:
    """
    Extracts the final boxed answer from a model's output string.

    Supports \boxed{}, integers, floats, fractions, and scientific notation.

    Args:
        text (str): The model output string.

    Returns:
        int, float, or None: Parsed numerical answer.
    """
    if not text:
        return None

    match = re.findall(r"\\boxed{(.*?)}", text)

    if not match:
        num_match = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?|[-+]?\d+/\d+", text)
        if not num_match:
            return None
        return float(num_match[-1]) if "." in num_match[-1] else int(num_match[-1])

    num_match = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?|[-+]?\d+/\d+", text)
    num_match = num_match[-1] if num_match else None

    num_str = match[-1].replace(",", "").strip()

    try:
        if "." in num_str or "e" in num_str:
            return float(num_str)
        if "/" in num_str:
            return float(Fraction(num_str))
        return int(num_str)
    except Exception:
        return float(num_match) if num_match else None


def compute_accuracy(
    predicted_values: List[Optional[Union[int, float]]], ground_truth_values: List[str]
) -> float:
    """
    Computes numeric accuracy using math.isclose.

    Args:
        predicted_values (list): Predicted values as floats or ints.
        ground_truth_values (list): Ground truth values as strings.

    Returns:
        float: Accuracy score between 0 and 1.
    """
    correct = sum(
        1
        for pred, truth in zip(predicted_values, ground_truth_values)
        if isinstance(pred, (int, float))
        and truth is not None
        and math.isclose(pred, float(truth.replace(",", "")), rel_tol=1e-6)
    )
    total = len(ground_truth_values)
    return correct / total if total > 0 else 0


def process_latex_answer(answer: Optional[str]) -> Optional[str]:
    """
    Strips LaTeX \text{} wrappers or whitespace from an answer.

    Args:
        answer (str): Answer string, possibly with LaTeX.

    Returns:
        str or None: Cleaned answer string.
    """
    if not answer:
        return None

    answer = str(answer).strip()

    text_match = re.match(r"\\text\{(.+?)\}", answer)
    if text_match:
        return text_match.group(1).strip()

    return answer


def general_compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compares text predictions with ground truths after normalization.

    Args:
        predictions (list): Model-generated answers.
        ground_truths (list): Reference answers.

    Returns:
        float: Accuracy score between 0 and 1.
    """
    correct = 0
    total = len(predictions)

    for pred, gt in zip(predictions, ground_truths):
        pred_processed = process_latex_answer(pred)
        gt_processed = process_latex_answer(gt)

        pred_normalized = str(pred_processed).strip().lower()
        gt_normalized = str(gt_processed).strip().lower()

        if pred_normalized == gt_normalized:
            correct += 1

    return correct / total if total > 0 else 0


# ============================
# Dataset Loader
# ============================


def load_dataset(
    dataset_name: str = "gsm8k",
    batch_size: int = 8,
    collate_fn: Callable = collate_fn,
    instruct: Optional[Callable[[Dict[str, Any]], Dict[str, str]]] = None,
) -> Tuple[Union[Dataset, IterableDataset], DataLoader]:
    """
    Loads and prepares a dataset with task-specific formatting.

    Args:
        dataset_name (str): Dataset identifier (e.g., 'gsm8k', 'HuggingFaceH4/MATH-500').
        batch_size (int): Batch size for the DataLoader.
        collate_fn (callable): Function to collate batches.
        instruct (callable, optional): Custom formatting function.

    Returns:
        tuple: (raw Hugging Face Dataset, PyTorch DataLoader)
    """
    if dataset_name == "gsm8k":
        dataset = datasets.load_dataset(dataset_name, "main", split="test")
        prepared_dataset = dataset.map(gsm8k_instruct, load_from_cache_file=False)

    elif dataset_name == "HuggingFaceH4/MATH-500":
        dataset = datasets.load_dataset(dataset_name, split="test")
        prepared_dataset = dataset.map(math500_instruct, load_from_cache_file=False)

    elif dataset_name == "lucasmccabe/logiqa/test":
        dataset = datasets.load_dataset("lucasmccabe/logiqa", split="test")
        prepared_dataset = dataset.map(logiqa_instruct, load_from_cache_file=False)

    elif dataset_name == "Maxwell-Jia/AIME_2024":
        dataset = datasets.load_dataset(dataset_name, split="train")
        prepared_dataset = dataset.map(aime2024_instruct, load_from_cache_file=False)

    elif dataset_name == "lucasmccabe/logiqa/train":
        dataset = datasets.load_dataset("lucasmccabe/logiqa", split="train")
        prepared_dataset = dataset.map(logiqa_instruct, load_from_cache_file=False)

    elif dataset_name == "lucasmccabe/logiqa/validation":
        dataset = datasets.load_dataset("lucasmccabe/logiqa", split="validation")
        prepared_dataset = dataset.map(logiqa_instruct, load_from_cache_file=False)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    prepared_dataset = prepared_dataset.map(
        lambda x: {"prompt_length": len(x["prompt"])},
        load_from_cache_file=False,
    )

    prepared_dataset = prepared_dataset.sort("prompt_length", reverse=True)

    dataloader = DataLoader(
        prepared_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return dataset, dataloader
