# scripts/format_dataset.py

from datasets import load_dataset, Dataset
import os
import json


def format_dolly(example):
    """Format instruction + context into a single prompt."""
    if example["context"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Context:\n{example['context']}"
    else:
        prompt = f"### Instruction:\n{example['instruction']}"

    return {
        "input_text": prompt,
        "target_text": example["response"]
    }


def get_formatted_dataset(save_path: str = None):
    # Load the Dolly dataset
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    # Apply formatting
    formatted_dataset = dataset.map(format_dolly)

    # Optionally save as JSON
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        formatted_dataset.to_json(save_path)
        print(f"✅ Saved formatted dataset to: {save_path}")

    return formatted_dataset


if __name__ == "__main__":
    save_path = "data/formatted_dolly.json"
    dataset = get_formatted_dataset(save_path)
    print("🔍 Sample:")
    print(json.dumps(dataset[0], indent=2))
