import math
import re
from fractions import Fraction

def extract_final_number(text):
    """
    Extracts the final boxed answer from a model's output string.
    
    Supports:
    - Answers wrapped in \boxed{}
    - Numbers with or without commas
    - Integers, floats, fractions, and scientific notation
    
    Args:
        text (str): The model output string.
    
    Returns:
        float or int or None: The final numerical answer extracted from the string.
    """
    
    if not text:
        return None

    # Look for the last occurrence of \boxed{...}
    match = re.findall(r"\\boxed{(.*?)}", text)

    if not match or match == []:
        # If no \boxed{} found, fallback to extracting any number in the string
        num_match = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?|[-+]?\d+/\d+", text)
        if not num_match:
            return None
        # Return last number as float if decimal or scientific, else int
        return float(num_match[-1]) if "." in num_match[-1] else int(num_match[-1])

    
    # Extract number inside the last \boxed{}
    num_match = re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?|[-+]?\d+/\d+", text)
    num_match = None if not num_match else num_match[-1]

    num_str = match[-1].replace(",", "").strip() # Clean up the boxed value

    try:
        if "." in num_str or "e" in num_str:
            return float(num_str)
        if "/" in num_str:
            return float(Fraction(num_str)) # Convert fraction to float
        return int(num_str)
    except:
        # Fallback to previously found number if parsing fails
        return float(num_match)


def compute_accuracy(predicted_values, ground_truth_values):
    """
    Compute accuracy between predicted and ground truth numerical values.
    
    Uses math.isclose() to handle float comparison with a small tolerance.
    
    Args:
        predicted_values (list): List of numbers predicted by the model.
        ground_truth_values (list): List of correct numbers (as strings).
    
    Returns:
        float: Accuracy score (between 0 and 1).
    """
    correct = sum(
        1
        for pred, truth in zip(predicted_values, ground_truth_values)
        if isinstance(pred, (int, float)) # Ensure prediction is numeric
        and truth is not None
        and math.isclose(pred, float(truth.replace(",", "")), rel_tol=1e-6) # Tolerant match
    )

    total = len(ground_truth_values)
    return correct / total if total > 0 else 0

def compute_metrics(tokenizer):
    """
    Returns a metrics function for Hugging Face Trainer evaluation.
    
    Args:
        tokenizer: The tokenizer used to decode model predictions and labels.
    
    Returns:
        function: A function that computes accuracy given eval_preds.
    """
    def metrics(eval_preds):
        preds, labels = eval_preds
        
        # Decode tokenized predictions and labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Import helper functions (assumes they exist in utils)
        from utils import extract_final_number, compute_accuracy
        
        # Extract final numeric answers from decoded strings
        pred_values = [extract_final_number(p) for p in decoded_preds]
        label_values = [extract_final_number(l) for l in decoded_labels]

        # Compute accuracy metric
        acc = compute_accuracy(pred_values, label_values)
        return {"accuracy": acc}
    
    return metrics