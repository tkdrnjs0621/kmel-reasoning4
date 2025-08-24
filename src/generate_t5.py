import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from tqdm import tqdm
import json

def main(args):
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    dataset = Dataset.from_json(args.dataset_path)

    with open(args.save_path, 'w') as out_f:
        for k in tqdm(dataset, desc="Processing"):
            instance = k['SMILES']
            # input_text = 
            input_text = f"Caption the following molecule: {instance}" if args.prompt else f"{instance}"

            text = tokenizer(input_text, return_tensors="pt").to('cuda')
            output = model.generate(input_ids=text["input_ids"], max_length=2048)
            output_text = tokenizer.decode(output[0].cpu())

            output_text = output_text.split(tokenizer.eos_token)[0]
            output_text = output_text.replace(tokenizer.pad_token, "")
            output_text = output_text.strip()

            k['prediction'] = output_text
            out_f.write(json.dumps(k) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions from SMILES strings using a transformer model.")
    parser.add_argument("--dataset_path", type=str, default='/home/tkdrnjs0621/work/kmel-reasoning/dataset/original/test.jsonl', help="Path to input JSONL file.")
    parser.add_argument("--save_path", type=str, default='/home/tkdrnjs0621/work/kmel-reasoning/dataset/test/baselines/out_molt5-large.jsonl', help="Path to output JSONL file.")
    parser.add_argument("--model_path", type=str, default="laituan245/molt5-large-smiles2caption", help="Pretrained model name or path.")
    parser.add_argument("--prompt", action='store_true')

    args = parser.parse_args()
    main(args)
