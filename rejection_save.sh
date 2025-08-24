#!/bin/bash

python3 /home/tkdrnjs0621/work/kmel-reasoning3/src/rejection_save.py \
  --input_data_path /home/tkdrnjs0621/work/kmel-reasoning3/dataset/original/hiv_train_2k.jsonl \
  --original_output_data_path /home/tkdrnjs0621/work/kmel-reasoning3/dataset/openai_batch_output/hiv_train_2k_output.jsonl \
  --original_batch_data_path /home/tkdrnjs0621/work/kmel-reasoning3/dataset/openai_batch/hiv_train_2k.jsonl  \
  --output_data_path /home/tkdrnjs0621/work/kmel-reasoning3/dataset/filtered_train_data/hiv_train_2k.jsonl \
  --input_column SELFIES \
  --id_column "id"
