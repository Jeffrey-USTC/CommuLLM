#!/usr/bin/env bash

cd /home/liuzj/code/LLaMA-Factory && \
export PATH="/home/liuzj/miniconda3/envs/llama-factory/bin:$PATH" && \
export CUDA_VISIBLE_DEVICES=0,1,2,3 && \
export ENABLE_MIDDLE_LAYER_CODEC=1 && \
export CODEC_LAYER_IDX=16 && \
export ACTIVATION_CODEC=uniform_4bit && \
export GRADIENT_CODEC=identity && \
export MASTER_PORT=29600 && \
llamafactory-cli train train_scripts/llama3_commonsense_sft.yaml 2>&1 | tee logs/llama3_sft_$(date +%Y%m%d_%H%M%S).log

