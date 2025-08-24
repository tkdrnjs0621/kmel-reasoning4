import json
import asyncio
import logging
from pathlib import Path
import argparse
import os
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI, APIError
from tqdm.asyncio import tqdm_asyncio

# --- Core Functions ---

async def get_llm_response(
    messages: List[Dict[str, str]],
    session: AsyncOpenAI,
    model_name: str,
) -> Optional[str]:
    try:
        response = await session.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
        )
        return response.choices[0].message.content
    except APIError as e:
        logging.error(f"API Error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during API call: {e}")
    return None

async def process_row(
    row: Dict[str, Any],
    session: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model_name: str,
    messages_key: str,
    output_path: str,
    save_per_row: bool,
):
    async with semaphore:
        messages = row.get(messages_key, [])
        if not messages:
            return None

        # Use all messages except the last one if it's from the assistant
        prompt_messages = messages
        if messages[-1].get("role") == "assistant":
            prompt_messages = messages[:-1]

        llm_output = await get_llm_response(prompt_messages, session, model_name)

        if llm_output is None:
            llm_output = "LLM_RESPONSE_FAILED"
            logging.error(f"Failed to get response for row: {row.get('id', 'N/A')}")

        result = row.copy()
        result["llm_response"] = llm_output

        if save_per_row:
            with open(output_path, "a") as f:
                f.write(json.dumps(result) + "\n")
        
        return result

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

    if not os.path.exists(args.input_path):
        logging.error(f"Input file not found: {args.input_path}")
        return

    with open(args.input_path, "r") as f:
        tasks_to_run = [json.loads(line) for line in f]

    logging.info(f"Found {len(tasks_to_run)} entries to process.")
    if not tasks_to_run:
        return

    # If not saving per row, clear the output file
    if not args.save_per_row and os.path.exists(args.output_path):
        os.remove(args.output_path)

    async_client = AsyncOpenAI(base_url=args.api_base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.semaphore_limit)

    async_tasks = [
        process_row(row, async_client, semaphore, args.model_name, args.messages_key, args.output_path, args.save_per_row)
        for row in tasks_to_run
    ]

    results = await tqdm_asyncio.gather(*async_tasks, desc="Sending requests to LLM")

    if not args.save_per_row:
        with open(args.output_path, "w") as f:
            for res in results:
                if res:
                    f.write(json.dumps(res) + "\n")
        logging.info(f"Saved all results to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a JSONL file with an LLM asynchronously.")
    
    parser.add_argument("--model_name", type=str, default="kmel", help="Name of the model to use.")
    parser.add_argument("--api_base_url", type=str, default="http://localhost:8010/v1/", help="API base URL for the LLM.")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key for the LLM.")
    parser.add_argument("--semaphore_limit", type=int, default=500, help="Concurrency limit for API requests.")
    
    parser.add_argument("--input_path", type=str, default="/home/tkdrnjs0621/work/kmel-reasoning4/dataset/chat/hiv_test_reasoning_chat.jsonl", help="Path to the input .jsonl file.")
    parser.add_argument("--output_path", type=str, default="result.jsonl", help="Path to save the output .jsonl file.")
    parser.add_argument("--log_file", type=str, default="api_request.log", help="File to write logs to.")
    
    parser.add_argument("--messages_key", type=str, default="messages", help="The key in the JSON object that contains the list of messages.")
    parser.add_argument("--save_per_row", action="store_true", help="Save the output for each row as it's processed.")

    args = parser.parse_args()
    asyncio.run(main(args))