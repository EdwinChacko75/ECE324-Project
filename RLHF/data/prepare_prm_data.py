# prepare_prm_data.py
import json
import argparse


def extract_labeled_steps(input_path, output_path):
    data = []
    with open(input_path, "r") as f:
        for line in f:
            item = json.loads(line)
            steps = item.get("label", {}).get("steps", [])
            for step in steps:
                completions = step.get("completions", [])
                for comp in completions:
                    text = comp.get("text")
                    rating = comp.get("rating")
                    if text is not None and rating is not None:
                        data.append({"input": text, "label": rating})
    with open(output_path, "w") as out:
        for entry in data:
            out.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Path to raw PRM Phase 2 .jsonl file"
    )
    parser.add_argument(
        "--output", required=True, help="Path to save cleaned output file"
    )
    args = parser.parse_args()
    extract_labeled_steps(args.input, args.output)
