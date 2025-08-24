
import os
import json
import argparse

def get_system_prompt(prompt_type):
    """Returns the system prompt string based on the selected type."""
    prompts = {
        "bace": "You are a chemical expert that performs Molecule property prediction task. You must check if the given molecule is the inhibitor of human beta-secretase 1 (BACE-1). If the given molecule can inhibit BACE-1, indicate via \"Yes\". Otherwise, response via \"No\". Do not add any auxiliary terms.",
        "hiv": "You are a chemical expert that performs Molecule property prediction task. You must check if the given molecule is the inhibitor of human immunodeficiency virus (HIV). If the given molecule can inhibit HIV replication, indicate via \"Yes\". Otherwise, response via \"No\". Do not add any auxiliary terms.",
        "bbbp": "You are a chemical expert that performs Molecule property prediction task. You must check if the given molecule has the blood-brain barrier permeability (BBBP). If the given molecule can penetrate the blood-brain barrier, indicate via \"Yes\". Otherwise, response via \"No\". Do not add any auxiliary terms.",
        "hiv_reasoning": "You are a chemical domain expert specializing in molecular property prediction.\nYou will be provided with a SELFIES representation of a molecule.\nYour task is to determine whether the given molecule functions as an inhibitor of the human immunodeficiency virus (HIV).\n\nAnalyze the molecule and provide a detailed explanation of your reasoning, citing relevant chemical properties, structural features, or known biological activity.\n\nAfter your reasoning, insert a new line.\n\nOn the following line, present your conclusion in the exact format of \"ANSWER: YES\", if the molecule can inhibit HIV replication, and \"ANSWER: NO\" if it cannot.",
        "default": "You are a helpful assistant.",
         "bbbp_reasoning": "You are a chemical domain expert specializing in molecular property prediction.\nYou will be provided with a SELFIES representation of a molecule.\nYour task is to determine whether the given molecule can penetrate the blood-brain barrier (BBBP).\n\nAnalyze the molecule and provide a detailed explanation of your reasoning, citing relevant chemical properties, structural features, physicochemical parameters (e.g., logP, molecular weight, polar surface area), or known biological activity that influence BBB permeability.\n\nAfter your reasoning, insert a new line.\n\nOn the following line, present your conclusion in the exact format of \"ANSWER: YES\" if the molecule can penetrate the blood-brain barrier, and \"ANSWER: NO\" if it cannot."
    }
    return prompts.get(prompt_type, prompts["default"])

def create_chat_dataset_for_file(input_file_path, system_prompt, prompt_type, input_column, result_column, output_path, if_test):
    """
    Processes a single JSONL file with a given system prompt to create a
    chat-formatted JSONL file.
    """
    if not os.path.exists(input_file_path):
        print(f"Error: The file '{input_file_path}' was not found. Skipping.")
        return

    processed_dir, selected_file = os.path.split(input_file_path)
    # base_filename = os.path.splitext(selected_file)[0]
    # output_filename = f"{base_filename}_{prompt_type}_chat.jsonl"
    # output_path = os.path.join(processed_dir, output_filename)

    print(f"Processing '{selected_file}' with prompt type '{prompt_type}'...")
    print(f"Output will be saved to '{output_path}'")

    try:
        with open(input_file_path, 'r') as infile, open(output_path, 'w') as outfile:
            for line in infile:
                try:
                    data = json.loads(line)
                    input_content = data.get(input_column)
                    result_content = data.get(result_column)

                    # if input_content is None or result_content is None:
                    #     print(f"Warning: Skipping line due to missing '{input_column}' or '{result_column}' key in: {line.strip()}")
                    #     continue

                    chat_data = {
                        "messages":  [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": input_content}
                        ] if if_test else [ 
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": input_content},
                            {"role": "assistant", "content": result_content}
                        ],
                        "SELFIES": data.get("SELFIES"),
                        "result" : data.get("result"),
                        "id" : data.get("id")
                    }
                    outfile.write(json.dumps(chat_data) + '\n')
                except json.JSONDecodeError:
                    print(f"Warning: Skipping line due to invalid JSON: {line.strip()}")
        print(f"Successfully created: {output_path}\n")
    except IOError as e:
        print(f"An error occurred during file processing for {input_file_path}: {e}\n")

def main():
    """
    Main function to configure and run the dataset creation process.
    """
    parser = argparse.ArgumentParser(description="Create a chat-formatted JSONL file from a given JSONL file.")
    parser.add_argument("--input_path", default="/home/tkdrnjs0621/work/kmel-reasoning3/dataset/filtered_train_data/hiv_train_10k.jsonl", help="The full path to the input JSONL file.")
    parser.add_argument("--prompt_type", default="hiv_reasoning", help="The type of system prompt to use. Defaults to 'default'.")
    parser.add_argument("--input_column", default="SELFIES", help="The name of the column containing the input data. Defaults to 'SELFIES'.")
    parser.add_argument("--type", default="train", help="The name of the column containing the result data. Defaults to 'result'.")
    parser.add_argument("--result_column", default="reasoning", help="The name of the column containing the result data. Defaults to 'result'.")
    parser.add_argument("--output_path", default="/home/tkdrnjs0621/work/kmel-reasoning3/dataset/processed_chat/hiv_train_reasoning_10k_chat.jsonl", help="The name of the column containing the result data. Defaults to 'result'.")

    args = parser.parse_args()

    system_prompt = get_system_prompt(args.prompt_type)
    create_chat_dataset_for_file(
        args.input_path,
        system_prompt,
        args.prompt_type,
        args.input_column,
        args.result_column,
        args.output_path,
        args.type=='test'
    )

    print("Processing complete.")

if __name__ == "__main__":
    main()
