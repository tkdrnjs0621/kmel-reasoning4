import json
import argparse
from datasets import load_dataset, Dataset
from tqdm import tqdm

def create_batch_file(dataset, output_filename, input_column, id_column, gt_given):
    
    system_prompt = """You are a chemical domain expert specializing in molecular property prediction.
You will be provided with a SELFIES representation of a molecule and a ground truth answer.
Your task is to give a detailed reasoning and determine if the given molecule functions as an inhibitor of the human immunodeficiency virus (HIV).

You must give an detailed explanation of the reasoning process. Use numbering and bold for each relevant point categories.
For every relevant point, provide a clear, concise paragraph explaining its role and significance in the context of potential HIV inhibition. 

The ground truth answer is given. The final conclusion must be identical to the ground truth, but you must not use or reference the ground truth explicitly while reasoning.
Do not write phrases such as “since the answer is…” or any variant that indicates reliance on the label.

After your reasoning, insert a new line.

On the following line, present your conclusion in the exact format of "ANSWER: YES", if the molecule can inhibit HIV replication, and "ANSWER: NO" if it cannot.
""".strip()

    jobs = []
    
    for row in tqdm(dataset):
        job = {
            "custom_id": row['id'],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4.1",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "SELFIES: "+row['SELFIES']+'\nGT Answer: '+row['result'][:-1].upper() if gt_given else row[input_column]}
                ],
                "temperature": 1,
                "top_p": 1,
            }
        }
        jobs.append(job)

    with open(output_filename, 'w') as f:
        for job in tqdm(jobs, desc=f"Writing {len(jobs)} jobs to {output_filename}"):
            f.write(json.dumps(job) + '\n')
            
    return len(jobs)

def main():
    """Main function to run the data generation process."""
    parser = argparse.ArgumentParser(description="Generate a batch.jsonl file for OpenAI API from the HiCUPID dataset.")
    parser.add_argument("--input_data_path", type=str, default="/home/tkdrnjs0621/work/kmel-reasoning3/dataset/original/hiv_train_2k.jsonl", help="Number of user_ids to process. Processes all users by default if not specified.")
    parser.add_argument("--output_data_path", type=str, default="/home/tkdrnjs0621/work/kmel-reasoning3/dataset/openai_batch/hiv_train_2k.jsonl", help="Number of user_ids to process. Processes all users by default if not specified.")
    parser.add_argument("--input_column", type=str, default="SELFIES", help="Number of user_ids to process. Processes all users by default if not specified.")
    parser.add_argument("--id_column", type=str, default="id", help="Number of user_ids to process. Processes all users by default if not specified.")
    parser.add_argument("--start_idx", type=int, default=0, help="Number of user_ids to process. Processes all users by default if not specified.")
    parser.add_argument("--end_idx", type=int, default=-1, help="Number of user_ids to process. Processes all users by default if not specified.")
    parser.add_argument("--gt_type", type=str, default="gt_given", help="Number of user_ids to process. Processes all users by default if not specified.")
    args = parser.parse_args()

    dataset = Dataset.from_json(args.input_data_path)
    dataset = dataset.select(range(args.start_idx,args.end_idx if args.end_idx>0 else len(dataset)))
    
    num_jobs_created = create_batch_file(dataset, args.output_data_path, args.input_column, args.id_column, args.gt_type=='gt_given')
    
    print(f"\nSuccessfully created batch.jsonl with {num_jobs_created} API requests.")

if __name__ == "__main__":
    main()