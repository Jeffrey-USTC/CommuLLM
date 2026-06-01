#!/usr/bin/env bash

cd /home/liuzj/code/LLaMA-Factory && \
export PATH="/home/liuzj/miniconda3/envs/llama-factory/bin:$PATH" && \
export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
export LLF_HOOK_ENABLE=1 && \
export LLF_HOOK_ACT=1 && \
export LLF_HOOK_QUANT_BITS=4 && \
export LLF_HOOK_GRAD=0 && \
export MASTER_PORT=29600 && \
llamafactory-cli train train_scripts/llama3_commonsense_sft.yaml 2>&1 | tee logs/llama3_sft_$(date +%Y%m%d_%H%M%S).log

