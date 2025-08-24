

import os
import json

def process_test_files():
    """
    Processes JSONL chat files to extract the assistant's response into a 'label'.

    This script reads '_chat.jsonl' files containing 'test' in their name
    from the source directory. For each line, it moves the assistant's
    response to a top-level 'label' field and removes it from the 'chat' list.
    The modified data is then saved to a new directory.
    """
    source_dir = '/home/tkdrnjs0621/work/kmel-reasoning2/dataset/processed_chat'
    output_dir = '/home/tkdrnjs0621/work/kmel-reasoning2/dataset/processed_chat_labeled'

    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found at '{source_dir}'")
        return

    if not os.path.exists(output_dir):
        print(f"Creating output directory at '{output_dir}'")
        os.makedirs(output_dir)

    try:
        files_to_process = [f for f in os.listdir(source_dir) if 'test' in f and f.endswith('_chat.jsonl')]
    except FileNotFoundError:
        print(f"Error: The directory '{source_dir}' was not found.")
        return

    if not files_to_process:
        print(f"No 'test' files ending with '_chat.jsonl' found in '{source_dir}'.")
        return

    print("Found the following test files to process:")
    for filename in files_to_process:
        print(f"- {filename}")
    print("-" * 20)


    for filename in files_to_process:
        input_path = os.path.join(source_dir, filename)
        output_filename = f"{os.path.splitext(filename)[0]}_labeled.jsonl"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Processing '{filename}'...")

        try:
            with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
                for line in infile:
                    try:
                        data = json.loads(line)
                        chat_list = data.get("message", [])
                        
                        assistant_response = None
                        # Find and remove the assistant message
                        for i in range(len(chat_list) - 1, -1, -1):
                            if chat_list[i].get("role") == "assistant":
                                assistant_response = chat_list.pop(i).get("content")
                                break
                        
                        if assistant_response is not None:
                            # Add the new 'label' key
                            data['label'] = assistant_response
                            outfile.write(json.dumps(data) + '\n')
                        else:
                            print(f"Warning: No assistant response found in line: {line.strip()}")

                    except json.JSONDecodeError:
                        print(f"Warning: Skipping line due to invalid JSON: {line.strip()}")
            
            print(f"Successfully created '{output_filename}'")

        except IOError as e:
            print(f"An error occurred during file processing for {filename}: {e}")

    print("\nAll processing complete.")
    print(f"Labeled files are saved in: {output_dir}")


if __name__ == "__main__":
    # First, let's create the directory that the user mentioned, so the script can find it.
    processed_chat_dir = '/home/tkdrnjs0621/work/kmel-reasoning2/dataset/processed_chat'
    if not os.path.exists(processed_chat_dir):
        print(f"Creating directory '{processed_chat_dir}' as it was mentioned but does not exist.")
        os.makedirs(processed_chat_dir)
        # As the directory was just created, we'll create a dummy file for the script to find.
        # This part can be removed if the directory and files are guaranteed to exist.
        dummy_file_path = os.path.join(processed_chat_dir, 'bace_test_prompt0_chat.jsonl')
        with open(dummy_file_path, 'w') as f:
            dummy_data = {
                "chat": [
                    {"role": "system", "content": "System prompt here"},
                    {"role": "user", "content": "[C][C]([C])([C])[C]"},
                    {"role": "assistant", "content": "Yes"}
                ]
            }
            f.write(json.dumps(dummy_data) + '\n')
        print(f"Created a dummy file for demonstration: {dummy_file_path}")


    process_test_files()
