import argparse
import json
import numpy as np
from transformers import AutoTokenizer
import pandas as pd

def analyze_token_counts(input_file, tokenizer_name, column_name):
    """
    Reads a jsonl file, calculates token counts for a specific column, and prints statistics.

    Args:
        input_file (str): Path to the input .jsonl file.
        tokenizer_name (str): Name of the Hugging Face tokenizer.
        column_name (str): The column to analyze from the jsonl file.
    """
    print(f"Loading tokenizer: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    token_counts = []
    print(f"Processing file: {input_file}")
    try:
        with open(input_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get(column_name)
                    if text is not None and isinstance(text, str):
                        tokens = tokenizer.encode(text)
                        token_counts.append(len(tokens))
                    else:
                        print(f"Warning: Column '{column_name}' not found or not a string in line: {line.strip()}")
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from line: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return

    if not token_counts:
        print("No data to analyze.")
        return

    # Calculate statistics using pandas
    stats = pd.Series(token_counts).describe(percentiles=[0, .25, .5, .75, 1])
    stats_dict = stats.to_dict()

    print("\n--- Token Count Statistics ---")
    print(f"Column analyzed: '{column_name}'")
    print(f"Total rows processed: {len(token_counts)}")
    print(f"avg ± std: {stats_dict.get('mean'):.2f} ± {stats_dict.get('std'):.2f}")
    print(
        f"q0/q1/q2/q3/q4: "
        f"{stats_dict.get('0%'):.0f} / "
        f"{stats_dict.get('25%'):.0f} / "
        f"{stats_dict.get('50%'):.0f} / "
        f"{stats_dict.get('75%'):.0f} / "
        f"{stats_dict.get('100%'):.0f}"
    )
    print("----------------------------\n")



def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze token counts in a jsonl file.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/home/tkdrnjs0621/work/kmel-reasoning3/output2.jsonl",
        help="Path to the input .jsonl file."
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Name of the Hugging Face tokenizer (e.g., 'bert-base-uncased')."
    )
    parser.add_argument(
        "--column_name",
        type=str,
        default="llm_output",
        help="The column name within the JSON objects to analyze."
    )
    args = parser.parse_args()

    analyze_token_counts(args.input_file, args.tokenizer_name, args.column_name)

if __name__ == "__main__":
    main()
