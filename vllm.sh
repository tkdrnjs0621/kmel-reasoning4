#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve saves/\
    --served_model_name kmel \
    --tensor-parallel-size 4 \
    --dtype bfloat16