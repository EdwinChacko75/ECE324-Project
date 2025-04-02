import json
import re

def load_jsonl(path):
    """Load a .jsonl file and return a list of JSON objects."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def extract_cleaned_fields(entry):
    try:
        # Extract question from the prompt
        q_match = re.search(r"Question:\s*(.*?)\nFinal Answer:", entry["prompt"], re.DOTALL)
        question = q_match.group(1).strip() if q_match else entry["prompt"]

        # Extract just the CoT explanation from the generated output
        generated = entry["generated_solution"]

        # Remove the prompt text and everything before the first actual step
        explanation_match = re.search(r"(Step\s*1:.*)", generated, re.DOTALL | re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else generated.strip()

        # Final numeric answer
        final_answer = str(entry.get("ground_truth_value") or entry.get("predicted_value")).strip()

        return {
            "question": question,
            "answer": explanation,
            "ground_truth": final_answer
        }
    except Exception as e:
        print(f"Skipping entry due to error: {e}")
        return None


def process_inference_file(input_path, output_path):
    raw_entries = load_jsonl(input_path)
    cleaned = [extract_cleaned_fields(entry) for entry in raw_entries if extract_cleaned_fields(entry)]

    # Save as JSONL for training
    with open(output_path, "w") as f:
        for item in cleaned:
            json.dump(item, f)
            f.write("\n")
    print(f"Saved cleaned dataset with {len(cleaned)} examples to {output_path}")

def validate_data(file_path="/home/ubuntu/reasonix/data/train.jsonl"):
    match_count = 0
    mismatch_count = 0
    total = 0

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            gt = str(data.get("ground_truth_value")).strip()
            pred = str(data.get("predicted_value")).strip()
            
            total += 1
            if gt == pred:
                match_count += 1
            else:
                mismatch_count += 1

    # Avoid division by zero
    if total > 0:
        match_pct = (match_count / total) * 100
        mismatch_pct = (mismatch_count / total) * 100
    else:
        match_pct = mismatch_pct = 0

    print(f"Total: {total}")
    print(f"Matches: {match_count} ({match_pct:.2f}%)")
    print(f"Mismatches: {mismatch_count} ({mismatch_pct:.2f}%)")

# Example usage
if __name__ == "__main__":
    input_jsonl = "/home/ubuntu/reasonix/data/deepseek.jsonl"      
    output_jsonl = "gsm8k_train_deepseek.jsonl"
    process_inference_file(input_jsonl, output_jsonl)
    validate_data(output_jsonl)