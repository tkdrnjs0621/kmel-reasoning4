#!/bin/bash

trap "echo 'SIGINT received. Killing all...'; kill 0; exit 1" SIGINT

python /home/tkdrnjs0621/work/kmel-reasoning3/src/rejection_save.py \
    --input_data_path /home/tkdrnjs0621/work/kmel-reasoning3/dataset/original/bbbp_train.jsonl \
    --original_output_data_path  /home/tkdrnjs0621/work/kmel-reasoning3/dataset/openai_batch_output/bbbp_train3_output.jsonl\
    --original_batch_data_path /home/tkdrnjs0621/work/kmel-reasoning3/dataset/openai_batch/bbbp_train3.jsonl\
    --rejected_output_data_path /home/tkdrnjs0621/work/kmel-reasoning3/dataset/openai_batch/bbbp_train4.jsonl \
    --output_data_path /home/tkdrnjs0621/work/kmel-reasoning3/dataset/filtered_train_data/bbbp_train3.jsonl \
    --input_column SELFIES\
    --id_column id&
sleep 0
    
wait