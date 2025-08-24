import json
import argparse
from datasets import load_dataset, Dataset
from tqdm import tqdm


def main():
    """Main function to run the data generation process."""
    parser = argparse.ArgumentParser(description="Generate a batch.jsonl file for OpenAI API from the HiCUPID dataset.")
    parser.add_argument("--input_data_path", type=str, default="/home/tkdrnjs0621/work/kmel-reasoning3/dataset/original/bbbp_train.jsonl", help="Number of user_ids to process. Processes all users by default if not specified.")
    parser.add_argument("--original_output_data_path", type=str, default="/home/tkdrnjs0621/work/kmel-reasoning3/dataset/openai_batch_output/bbbp_train_output.jsonl", help="Number of user_ids to process. Processes all users by default if not specified.")
    parser.add_argument("--original_batch_data_path", type=str, default="/home/tkdrnjs0621/work/kmel-reasoning3/dataset/openai_batch/bbbp_train.jsonl", help="Number of user_ids to process. Processes all users by default if not specified.")
    
    parser.add_argument("--rejected_output_data_path", type=str, default="/home/tkdrnjs0621/work/kmel-reasoning3/dataset/openai_batch/bbbp_train2.jsonl", help="Number of user_ids to process. Processes all users by default if not specified.")
    parser.add_argument("--output_data_path", type=str, default="/home/tkdrnjs0621/work/kmel-reasoning3/dataset/filtered_train_data/bbbp_train.jsonl", help="Number of user_ids to process. Processes all users by default if not specified.")
    parser.add_argument("--input_column", type=str, default="SELFIES", help="Number of user_ids to process. Processes all users by default if not specified.")
    parser.add_argument("--id_column", type=str, default="id", help="Number of user_ids to process. Processes all users by default if not specified.")
    parser.add_argument("--save_rejected", action='store_true', help="Number of user_ids to process. Processes all users by default if not specified.")
    args = parser.parse_args()

    dataset = Dataset.from_json(args.input_data_path)
    dataset_dict = {}
    for k in dataset:
        dataset_dict[k['id']]=k
    
    dataset_out = Dataset.from_json(args.original_output_data_path)
    dataset_batch = Dataset.from_json(args.original_batch_data_path)
    dataset_batch_dict = {}
    for k in dataset_batch:
        dataset_batch_dict[k['custom_id']]=k


    final_list = []
    rejected_list = []
    for k in dataset_out:
        pred_true = 'yes' in k['response']['body']['choices'][0]['message']['content'].lower().split('answer:')[-1].strip()
        gt_true = dataset_dict[k['custom_id']]['result']=='Yes.'
        if(pred_true==gt_true):
            tmp = dataset_dict[k['custom_id']].copy()
            tmp['reasoning'] = k['response']['body']['choices'][0]['message']['content']
            final_list.append(tmp)
        else:
            rejected_list.append(dataset_batch_dict[k['custom_id']])

    Dataset.from_list(final_list).to_json(args.output_data_path)
    if(args.save_rejected):
        Dataset.from_list(rejected_list).to_json(args.rejected_output_data_path)

if __name__ == "__main__":
    main()