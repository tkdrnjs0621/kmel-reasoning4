from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
import os
from datasets import Dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import torch
import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Run")
    parser.add_argument("--model_path", type=str, default="/home/tkdrnjs0621/ssd1/tkdrnjs0621/kmel_ckpts_backslash/llama3.1-8b/sft_full", help="model name for evaluation")
    parser.add_argument("--dataset_path", type=str, default="/home/tkdrnjs0621/work/newkmel/dataset/test/test-zs-vanilla.jsonl", help="model name for evaluation")
    parser.add_argument("--save_path", type=str, default="newnewnewrs.jsonl", help="model name for evaluation")
    parser.add_argument("--wo_think", action='store_true', help="model name for evaluation")

    args = parser.parse_args()

    llm = LLM(
        model=args.model_path, 
        gpu_memory_utilization=0.9, 
        enable_chunked_prefill=True,
        trust_remote_code=True
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=8192) #if args.wo_think else SamplingParams(temperature=0, max_tokens=8192) 

    dataset = Dataset.from_json(args.dataset_path)

    with open(args.save_path, 'w', encoding='utf-8') as f:
        # for k in tqdm(dataset):
            
        #     output = llm.chat(
        #         k['messages'],
        #         sampling_params
        #         ,chat_template_kwargs={"enable_thinking": False}
        #     )[0].outputs[0].text if args.wo_think else llm.chat(
        #         k['messages'],
        #         sampling_params
        #     )[0].outputs[0].text

        #     k['prediction'] = output
        #     f.write(json.dumps(k, ensure_ascii=False) + '\n')

        batch_size = 16  # Adjust based on your GPU memory
        data_list = dataset.to_list()
        for i in tqdm(range(0, len(data_list), batch_size)):
            batch = data_list[i:i+batch_size]
            messages_batch = [item['messages'] for item in batch]

            outputs = llm.chat(messages_batch, sampling_params)

            for j, output in enumerate(outputs):
                batch[j]['prediction'] = output.outputs[0].text
                f.write(json.dumps(batch[j], ensure_ascii=False) + '\n')