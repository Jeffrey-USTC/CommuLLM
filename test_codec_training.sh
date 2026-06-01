#!/bin/bash
# 小规模训练测试脚本 - 验证 CodecBoundary 集成

export CUDA_VISIBLE_DEVICES=0

# 启用 codec
export ENABLE_MIDDLE_LAYER_CODEC=1
export CODEC_LAYER_IDX=16
export ACTIVATION_CODEC_TYPE=identity
export GRADIENT_CODEC_TYPE=identity

# 训练参数
MODEL_PATH="meta-llama/Meta-Llama-3-8B"
DATA_PATH="yahma/alpaca-cleaned"
OUTPUT_DIR="./test_codec_output"

# 清理旧输出
rm -rf ${OUTPUT_DIR}

# 运行训练 (只训练 10 步用于测试)
/home/liuzj/miniconda3/envs/llama-factory/bin/python -m llamafactory.cli train \
    --stage sft \
    --do_train \
    --model_name_or_path ${MODEL_PATH} \
    --dataset alpaca_en \
    --template llama3 \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 512 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --warmup_steps 0 \
    --save_steps 100 \
    --eval_steps 100 \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --max_steps 10 \
    --val_size 0 \
    --plot_loss \
    --bf16

echo "Training completed. Check logs in ${OUTPUT_DIR}"
