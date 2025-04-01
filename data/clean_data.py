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

# Example usage
if __name__ == "__main__":
    input_jsonl = "/home/ubuntu/reasonix/data/train.jsonl"      
    output_jsonl = "cot_cleaned_dataset.jsonl"
    process_inference_file(input_jsonl, output_jsonl)
