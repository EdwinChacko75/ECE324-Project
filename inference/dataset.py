import datasets
from torch.utils.data import DataLoader
import re
from fractions import Fraction
import math


def prepare_example(example):
    """
    Prepare a single example for inference:
      - Extract the math problem as the question.
      - Extract the detailed ground truth chain-of-thought.
      - Parse out the final answer (ground truth value) which is preceded by "\boxed{".
      - Format the prompt for the LLM.
    """
    question = example.get("question")
    ground_truth = example.get("answer")

    prompt = (
        f"Question: {question}\n"
        "Answer: Please solve the problem and provide your answer in a detailed explanation. "
        "Write the final numerical answer inside \\boxed{} at the end."
    )

    ground_truth_value = None
    if ground_truth and "####" in ground_truth:
        ground_truth_value = ground_truth.split("####")[-1].strip()

    return {
        "prompt": prompt,
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
    correct = sum(
        1
        for pred, truth in zip(predicted_values, ground_truth_values)
        if isinstance(pred, (int, float))
        and truth is not None
        and math.isclose(pred, float(truth.replace(",", "")), rel_tol=1e-6)
    )

    total = len(ground_truth_values)
    return correct / total if total > 0 else 0


def load_dataset(dataset_name="gsm8k", batch_size=8, collate_fn=collate_fn):
    """
    Get the dataloader for a specified dataset.
    """
    # Load dataset and prepare it
    dataset = datasets.load_dataset(dataset_name, "main", split="test")

    prepared_dataset = dataset.map(prepare_example)

    prepared_dataset = prepared_dataset.map(
        lambda x: {"prompt_length": len(x["prompt"])}
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


# This is testing of the accuracy computation. Sanity check before running a new model.
if __name__ == "__main__":
    test_cases = [
        # Standard boxed format
        "The answer is \\boxed{42}.",
        """Question: While playing with her friends in their school playground, Katelyn saw 50 fairies flying above the nearby forest. After about twenty minutes, one of her friends saw half as many fairies as Katelyn saw come from the east and join the fairies that were there. Ten minutes later, 30 fairies flew away. How many fairies are remaining?
Answer: Please solve the problem and provide your answer in a detailed explanation. Ensure that your final answer is provided on a new line starting with '####' followed by the final numerical value.assistant

#### Step 1: Determine the initial number of fairies Katelyn saw.
Katelyn saw 50 fairies initially.

#### Step 2: Calculate the number of fairies that joined Katelyn's group.
One of her friends saw half as many fairies as Katelyn saw come from the east and join the fairies that were there. This means 50 / 2 = 25 fairies joined Katelyn's group.

#### Step 3: Calculate the total number of fairies present after 20 minutes.
50 (initial fairies) + 25 (fairies that joined) = 75 fairies.

#### Step 4: Calculate the number of fairies that flew away.
30 fairies flew away.

#### Step 5: Calculate the number of fairies remaining.
75 (total fairies present) - 30 (fairies that flew away) = 45 fairies.

#### Step 6: Provide the final answer in the required format.
#### \boxed{45}""",
        # Extra spaces inside boxed
        "After solving, we get: \\boxed{ 3.14 }.",
        # Negative number
        "Thus, the final answer is \\boxed{-27}.",
        # Large number with comma (should be parsed as an integer)
        "The final value is \\boxed{1,234}.",
        # Scientific notation
        "Using calculations, we find \\boxed{2.5e3}.",
        # Incorrectly formatted box (should not extract anything)
        "The result is \\box{100}.",
        "The result is 1,002.",
        # No box present
        "I think the answer is -256.38, but I'm not sure.",
        # Boxed appears mid-text
        "We first calculate 10 + 5 = 15. Then, we have \\boxed{15}. The method used was...",
        # Boxed with text inside (should not match)
        "Finally, we get \\boxed{the answer is 99}.",
        # Multiple boxed answers (should return the last one)
        "We first compute \\boxed{12}. However, the final answer is \\boxed{24}.",
        "We first compute \\boxed{12}. However, the final answer is \\boxed{24/2}.",
    ]
    truth = ["24,000" for x in test_cases]
    pred = [extract_final_number(x) for x in test_cases]
    for i, test in enumerate(test_cases):
        extracted = extract_final_number(test)
        print(f"Test {i+1}: Extracted -> {extracted}")

    truth.append("24000")
    pred.append(24000)
    print(truth)
    print(pred)
    print(compute_accuracy(pred, truth))
