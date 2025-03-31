import math
import re
from fractions import Fraction

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
    correct = sum(
        1
        for pred, truth in zip(predicted_values, ground_truth_values)
        if isinstance(pred, (int, float))
        and truth is not None
        and math.isclose(pred, float(truth.replace(",", "")), rel_tol=1e-6)
    )

    total = len(ground_truth_values)
    return correct / total if total > 0 else 0

def compute_metrics(tokenizer):
    def metrics(eval_preds):
        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        from utils import extract_final_number, compute_accuracy
        pred_values = [extract_final_number(p) for p in decoded_preds]
        label_values = [extract_final_number(l) for l in decoded_labels]

        acc = compute_accuracy(pred_values, label_values)
        return {"accuracy": acc}
    
    return metrics