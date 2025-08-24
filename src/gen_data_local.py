import json
import asyncio
import logging
from pathlib import Path
import argparse
import os
from typing import Dict, Any, Optional
from openai import AsyncOpenAI, APIError
from tqdm.asyncio import tqdm_asyncio

# --- Prompts (as requested by user) ---
PROMPT_TEMPLATES = {
    "default":"""You are a chemical domain expert specializing in molecular property prediction.
You will be provided with a SELFIES representation of a molecule.
Your task is to determine whether the given molecule can penetrate the blood-brain barrier (BBBP).

Analyze the molecule and provide a detailed explanation of your reasoning, citing relevant chemical properties, structural features, physicochemical parameters (e.g., logP, molecular weight, polar surface area), or known biological activity that influence BBB permeability.

After reasoning, present your conclusion in the format of \"ANSWER: YES\", if the molecule can penetrate the blood-brain barrier, and \"ANSWER: NO\" if it cannot, with out using any bold fonts.
SELFIES: {selfies}
""",
    "hiv": """You are a chemical domain expert specializing in molecular property prediction.
You will be provided with a SELFIES representation of a molecule.
Your task is to determine whether the given molecule functions as an inhibitor of the human immunodeficiency virus (HIV).

Analyze the molecule and provide a detailed explanation of your reasoning, citing relevant chemical properties, structural features, or known biological activity.

Present your conclusion in the exact format of \"ANSWER: YES\", if the molecule can inhibit HIV replication, and \"ANSWER: NO\" if it cannot, with out using any bold fonts.
SELFIES: {selfies}
"""
}

# --- Core Functions ---

async def get_llm_response(
    prompt: str,
    session: AsyncOpenAI,
    model_name: str,
) -> Optional[str]:
    try:
        messages = [{"role": "user", "content": prompt}]
        response = await session.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=100000,
            reasoning_effort='high',
            temperature=0.8,
        )
        return response.choices[0].message.content
    except APIError as e:
        logging.error(f"API Error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during API call: {e}")
    return None

async def process_and_update_item(
    task_info: Dict[str, Any],
    session: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model_name: str,
    output_file: str,
    lock: asyncio.Lock,
):
    async with semaphore:
        prompt = task_info["prompt"]
        expected_result = task_info["result"]
        item_id = task_info["id"]

        llm_output = None
        attempt = 0
        while True:
            attempt += 1
            llm_output = await get_llm_response(prompt, session, model_name)

            if llm_output and expected_result.lower().split('.')[0] in llm_output.lower().split('answer:')[-1].strip():
                logging.info(f"Correct answer received for {item_id} on attempt {attempt}.")
                break
            
            logging.warning(f"Incorrect answer for {item_id} (attempt {attempt}). LLM output: {llm_output}. Expected: {expected_result}")
            await asyncio.sleep(1) # Wait for 1 second before retrying

        if llm_output is None:
            llm_output = "LLM_RESPONSE_FAILED"
            logging.error(f"Failed to get response for item: {item_id}")
        
        output_data = {
            "id": item_id,
            "llm_output": llm_output,
            "expected_result": expected_result,
            "prompt": prompt,
        }
        
        async with lock:
            with open(output_file, "a") as f:
                f.write(json.dumps(output_data) + "\n")

async def main(args):
    # --- Logging Setup ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file, mode='w'),
            logging.StreamHandler()
        ]
    )

    tasks_to_run = []
    input_files = []
    if os.path.isdir(args.input_path):
        for ext in ('*.json', '*.jsonl'):
            input_files.extend(Path(args.input_path).rglob(ext))
    else:
        input_files.append(Path(args.input_path))

    prompt_template = PROMPT_TEMPLATES.get(args.prompt_name, PROMPT_TEMPLATES["default"])

    for file_path in input_files:
        with open(file_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    prompt_text = data.get(args.prompt_key)
                    result = data.get("result")
                    item_id = data.get("id")
                    if prompt_text and result and item_id:
                        tasks_to_run.append({
                            "prompt": prompt_template.format(selfies=prompt_text),
                            "input_file": str(file_path),
                            "result": result,
                            "id": item_id,
                        })
                except (json.JSONDecodeError, KeyError) as e:
                    logging.error(f"Skipping line in {file_path} due to error: {e}")
                    continue

    logging.info(f"Found {len(tasks_to_run)} entries to process.")
    if not tasks_to_run:
        return

    lock = asyncio.Lock()

    if os.path.exists(args.output_file):
        os.remove(args.output_file)
        
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    async_client = AsyncOpenAI(base_url=args.api_base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.semaphore_limit)

    async_tasks = [
        process_and_update_item(task_info, async_client, semaphore, args.model_name, args.output_file, lock)
        for task_info in tasks_to_run
    ]

    await tqdm_asyncio.gather(*async_tasks, desc="Sending requests to LLM")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process text files with an LLM asynchronously with rejection sampling.")
    
    parser.add_argument("--model_name", type=str, default="gpt-oss-120b", help="Name of the model to use.")
    parser.add_argument("--api_base_url", type=str, default="http://localhost:8000/v1/", help="API base URL for the LLM.")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key for the LLM.")
    parser.add_argument("--semaphore_limit", type=int, default=200, help="Concurrency limit for API requests.")
    
    parser.add_argument("--input_path", type=str, default="/home/tkdrnjs0621/work/kmel-reasoning3/dataset/original/bbbp_train.jsonl", help="Path to an input file or directory containing .json/.jsonl files.")
    parser.add_argument("--output_file", type=str, default="output2.jsonl", help="Path to save the output jsonl file.")
    parser.add_argument("--log_file", type=str, default="api_request.log", help="File to write logs to.")
    
    parser.add_argument("--prompt_name", type=str, default="default", help="Name of the prompt template to use.")
    parser.add_argument("--prompt_key", type=str, default="SELFIES", help="The key in the JSON to use for the prompt's input.")

    args = parser.parse_args()
    asyncio.run(main(args))
