from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from datasets import Dataset
from tqdm import tqdm
import torch
import argparse
import json

def apply_chat_template_internLM(input_ls):
    txt=""
    for d in input_ls:
        role = d['role']
        content = d['content']
        txt+=f"""<|im_start|>{role}\n{content}<|im_end|>\n"""
    txt+="<|im_start|>assistant\n"
    return txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument("--model_path", type=str, default="/home/tkdrnjs0621/work/dsail-k-melloddy/reasoning/LLaMA-Factory/saves/llama3.1-8b/sft_full", help="model name for evaluation")
    parser.add_argument("--dataset_path", type=str, default="/home/tkdrnjs0621/work/newkmel/dataset/test/test-zs-vanilla.jsonl", help="path to dataset")
    parser.add_argument("--save_path", type=str, default="newnewnewrs.jsonl", help="output save path")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).to(device)
    model.eval()

    dataset = Dataset.from_json(args.dataset_path)

    with open(args.save_path, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset):
            # prompt = tokenizer.apply_chat_template(
            #     example["messages"],
            #     tokenize=False,
            #     add_generation_prompt=True
            # )
            prompt = apply_chat_template_internLM(example["messages"])

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            output_text = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            example['prediction'] = output_text.strip()
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
