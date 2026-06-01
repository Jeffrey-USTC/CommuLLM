#!/usr/bin/env bash

# 单独评估脚本：只用 GPU 0,1，评估已训练好的 adapter

set +e  # 单个数据集失败不中断整体

cd /home/liuzj/code/LLaMA-Factory

# =====================================================================
# 配置区：修改这里指定要评估的 adapter 路径
# =====================================================================
ADAPTER_DIR="/home/liuzj/code/LLaMA-Factory/runs/LLaMA3-8B/lora/codec_layer16_identity_20260531_230312/checkpoint-31900"

# =====================================================================
# 评估环境配置
# =====================================================================

# 评估使用 dora conda 环境的 Python
EVAL_PYTHON_BIN="/home/liuzj/miniconda3/envs/dora/bin/python"

# 测试数据集根目录
export COMMONSENSE_TEST_DATA_ROOT="/home/liuzj/data/datasets/commonsense_170k/test_data"
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

# 训练为 LoRA，评估同样按 LoRA 加载
EVAL_ADAPTER="LoRA"
LORA_WEIGHTS="$ADAPTER_DIR"
OUTPUT_DIR="$ADAPTER_DIR"

# 基础模型路径
BASE_MODEL="/home/liuzj/data/models/LLM-Research/Meta-Llama-3-8B"

# DoRA commonsense_reasoning 脚本目录
EVAL_SCRIPT_DIR="/home/liuzj/code/DoRA/commonsense_reasoning"

# 只用 GPU 0,1（避免与 GPU 2,3 上的其他任务冲突）
EVAL_GPUS="0,1"
export CUDA_VISIBLE_DEVICES="$EVAL_GPUS"

# 将 GPU 字符串转换为数组
IFS=',' read -ra GPU_ARRAY <<< "$EVAL_GPUS"

echo "=========================================="
echo "开始并行评估（仅使用 GPU 0,1）"
echo "Base Model:   $BASE_MODEL"
echo "LoRA Weights: $LORA_WEIGHTS"
echo "Output Dir:   $OUTPUT_DIR"
echo "Eval Script:  $EVAL_SCRIPT_DIR"
echo "=========================================="

# 手动分配任务到 GPU 0 和 GPU 1
# GPU 0: boolq, winogrande, piqa, ARC-Easy
# GPU 1: social_i_qa, ARC-Challenge, openbookqa, hellaswag
declare -a GPU0_TASKS=("boolq" "winogrande" "piqa" "ARC-Easy")
declare -a GPU1_TASKS=("social_i_qa" "ARC-Challenge" "openbookqa" "hellaswag")

# 在指定GPU上串行执行任务列表
run_gpu_tasks() {
    local gpu_id=$1
    shift
    local tasks=("$@")

    for dataset in "${tasks[@]}"; do
        echo "启动评估任务: $dataset 在 GPU $gpu_id"
        cd "$EVAL_SCRIPT_DIR" && \
        CUDA_VISIBLE_DEVICES=$gpu_id $EVAL_PYTHON_BIN commonsense_evaluate2.py \
            --model LLaMA3-8B \
            --adapter $EVAL_ADAPTER \
            --dataset $dataset \
            --base_model "$BASE_MODEL" \
            --batch_size 1 \
            --lora_weights "$LORA_WEIGHTS" > "$OUTPUT_DIR/$dataset.txt" 2>&1
        echo "GPU $gpu_id 完成任务: $dataset (退出码: $?)"
    done
    echo "GPU $gpu_id 所有任务执行完成"
}

# 并行启动 GPU 0 和 GPU 1 的任务（每个GPU内部串行执行）
if [ ${#GPU0_TASKS[@]} -gt 0 ]; then
    run_gpu_tasks ${GPU_ARRAY[0]} "${GPU0_TASKS[@]}" &
fi
if [ ${#GPU1_TASKS[@]} -gt 0 ]; then
    run_gpu_tasks ${GPU_ARRAY[1]} "${GPU1_TASKS[@]}" &
fi

# 等待所有后台任务完成
echo "等待所有评估任务完成..."
wait
echo "所有评估任务已完成！"

# 汇总所有数据集的准确率统计
echo ""
echo "=========================================="
echo "所有数据集评估完成，汇总统计如下："
echo "=========================================="
for dataset_file in "$OUTPUT_DIR"/*.txt; do
    if [ -f "$dataset_file" ]; then
        dataset_name=$(basename "$dataset_file" .txt)
        accuracy=$(grep "^${dataset_name}:" "$dataset_file" | tail -1 | awk -F': ' '{print $2}')
        if [ ! -z "$accuracy" ]; then
            printf "%-20s: %s\n" "$dataset_name" "$accuracy"
        else
            echo "$dataset_name: 未找到准确率（可能失败）"
        fi
    fi
done
echo "=========================================="
echo ""
