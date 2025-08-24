#!/bin/bash

trap "echo 'SIGINT received. Killing all...'; kill 0; exit 1" SIGINT

CUDA_VISIBLE_DEVICES=3 python /home/tkdrnjs0621/work/kmel-reasoning3/src/generate.py \
    --model_path /home/tkdrnjs0621/work/kmel-reasoning3/LLaMA-Factory/saves/qwen3-8b/hiv_sft_full \
    --dataset_path /home/tkdrnjs0621/work/kmel-reasoning3/dataset/chat/hiv_test_reasoning_chat.jsonl \
    --save_path /home/tkdrnjs0621/work/kmel-reasoning3/result/hiv_reasoning.jsonl &
sleep 0
    
wait