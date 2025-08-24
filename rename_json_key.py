import os
import json
import shutil

def rename_key_in_file(file_path, old_key, new_key):
    """
    Renames a key in each JSON object within a single JSON or JSONL file.
    Handles both single-object .json files and line-delimited .jsonl files.
    """
    temp_file_path = file_path + '.tmp'
    replacements_count = 0
    is_jsonl = file_path.endswith('.jsonl')

    try:
        with open(file_path, 'r') as infile, open(temp_file_path, 'w') as outfile:
            if is_jsonl:
                for line in infile:
                    try:
                        data = json.loads(line)
                        if old_key in data:
                            data[new_key] = data.pop(old_key)
                            replacements_count += 1
                        outfile.write(json.dumps(data) + '\n')
                    except json.JSONDecodeError:
                        # Write invalid lines back as they were
                        outfile.write(line)
                        print(f"Warning: Skipping invalid JSON line in {os.path.basename(file_path)}: {line.strip()}")
            else:  # Handle as a single JSON object file
                try:
                    data = json.load(infile)
                    if isinstance(data, dict) and old_key in data:
                        data[new_key] = data.pop(old_key)
                        replacements_count += 1
                    # Pretty print the output for single JSON files
                    json.dump(data, outfile, indent=4)
                except json.JSONDecodeError:
                    print(f"Error: Could not parse single JSON object file: {os.path.basename(file_path)}. It will be skipped.")
                    # If parsing fails, abort for this file by removing the temp file
                    os.remove(temp_file_path)
                    return

        # If we've made it here, processing was successful. Replace the original file.
        shutil.move(temp_file_path, file_path)
        if replacements_count > 0:
            print(f"Processed '{os.path.basename(file_path)}': Renamed {replacements_count} key(s).")
        else:
            print(f"Processed '{os.path.basename(file_path)}': No keys named '{old_key}' found.")

    except (IOError, json.JSONDecodeError) as e:
        print(f"An error occurred while processing {os.path.basename(file_path)}: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    except Exception as e:
        print(f"An unexpected error occurred with {os.path.basename(file_path)}: {e}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def main():
    """
    Main function to configure and run the key renaming process for a directory.
    """
    # --- Configuration ---
    # 1. Specify the full path to the directory you want to process.
    target_directory = '/home/tkdrnjs0621/work/kmel-reasoning2/dataset/processed_chat'

    # 2. Define the key to be renamed.
    old_key = "message"

    # 3. Define the new key name.
    new_key = "messages"
    # --- End of Configuration ---

    if not os.path.isdir(target_directory):
        print(f"Error: The directory '{target_directory}' was not found.")
        return

    print(f"Scanning directory: {target_directory}")
    print(f"Will rename key '{old_key}' to '{new_key}' in all .json and .jsonl files.\n")

    for filename in os.listdir(target_directory):
        if filename.endswith('.json') or filename.endswith('.jsonl'):
            file_path = os.path.join(target_directory, filename)
            rename_key_in_file(file_path, old_key, new_key)

    print("\nAll processing complete.")

if __name__ == "__main__":
    main()