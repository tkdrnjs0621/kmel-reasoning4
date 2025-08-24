#!/bin/bash

trap "echo 'SIGINT received. Killing all...'; kill 0; exit 1" SIGINT

python create_chat_jsonl.py \
    --input_path /home/tkdrnjs0621/work/kmel-reasoning3/dataset/filtered_train_data/hiv_train_2k.jsonl  \
    --prompt_type hiv_reasoning\
    --input_column SELFIES \
    --type train \
    --result_column reasoning \
    --output_path /home/tkdrnjs0621/work/kmel-reasoning3/dataset/chat/hiv_train_2k_reasoning_chat.jsonl &
sleep 0
    
wait