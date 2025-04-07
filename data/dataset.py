# dataset.py
import re
import math
from typing import Optional, Union, List, Tuple, Dict, Any
from fractions import Fraction

import datasets
from datasets import Dataset
from torch.utils.data import DataLoader


def prepare_example(example: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Prepare a single GSM8K example for inference.

    - Extracts the question and full ground truth answer.
    - Parses the final numeric answer (after '####').
    - Formats a structured prompt requesting a natural language explanation.

    Args:
        example (dict): A GSM8K dataset example with 'question' and 'answer'.

    Returns:
        dict: A dictionary containing:
            - 'prompt': Formatted input for the LLM.
            - 'ground_truth': Original answer field.
            - 'ground_truth_value': Final numeric answer as a string (if available).
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


def collate_fn(batch: List[Dict[str, Optional[str]]]) -> Dict[str, List[Optional[str]]]:
    """
    Collate a batch of GSM8K examples into separate lists for inference.

    Args:
        batch (list): A list of prepared examples.

    Returns:
        dict: A dictionary with lists of prompts, ground_truths, and ground_truth_values.
    """
    prompts = [item["prompt"] for item in batch]
    ground_truths = [item["ground_truth"] for item in batch]
    ground_truth_values = [item["ground_truth_value"] for item in batch]

    return {
        "prompts": prompts,
        "ground_truths": ground_truths,
        "ground_truth_values": ground_truth_values,
    }


def extract_final_number(text: str) -> Optional[Union[int, float]]:
    """
    Extracts the final boxed numerical answer from a model's output string.

    Handles:
    - \boxed{} wrapping
    - Decimals, scientific notation, fractions
    - Fallback to last numeric value if no \boxed{} is found

    Args:
        text (str): Model-generated string output.

    Returns:
        float | int | None: Extracted number or None if not found.
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
    Computes numerical accuracy using strict float comparison with tolerance.

    Args:
        predicted_values (list): Model-predicted numerical values.
        ground_truth_values (list): Reference numerical answers (as strings).

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
    Removes LaTeX formatting (e.g., \\text{}) from answer strings.

    Args:
        answer (str): Raw LaTeX-style answer string.

    Returns:
        str or None: Cleaned answer.
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
    Computes string-based accuracy with normalized comparison.

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


def load_dataset(
    dataset_name: str = "gsm8k",
    batch_size: int = 8,
    split: Optional[str] = "test",
    collate_fn: Callable[
        [List[Dict[str, Optional[str]]]], Dict[str, List[Optional[str]]]
    ] = collate_fn,
) -> Tuple[Dataset, DataLoader]:
    """
    Loads and prepares the GSM8K dataset for evaluation.

    Args:
        dataset_name (str): Dataset name (must be 'gsm8k').
        batch_size (int): Batch size for DataLoader.
        split (str): Dataset split to load ('train', 'test', etc.).
        collate_fn (callable): Function to collate batches.

    Returns:
        tuple: Raw dataset and corresponding DataLoader.
    """
    if dataset_name != "gsm8k":
        raise ValueError(f"Only 'gsm8k' is supported, got: {dataset_name}")

    dataset = datasets.load_dataset(dataset_name, "main", split=split)
    prepared_dataset = dataset.map(prepare_example, load_from_cache_file=False)

    # Compute and sort by prompt length (longest first)
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
