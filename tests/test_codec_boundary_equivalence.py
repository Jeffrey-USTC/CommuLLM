"""
测试 CodecBoundary 的数值等价性

验证目标:
1. Identity codec(激活和梯度都不修改)下,新实现与"完全不用 hook"的 baseline 数值完全一致
2. 验证 loss 和梯度的 bit-exact 等价性(BF16 精度下)

运行方式:
    cd /home/liuzj/code/LLaMA-Factory
    /home/liuzj/miniconda3/envs/dora/bin/python tests/test_codec_boundary_equivalence.py
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class SimpleLlamaLayer(nn.Module):
    """简化的 LLaMA decoder layer,用于测试"""

    def __init__(self, hidden_size=128, intermediate_size=256):
        super().__init__()
        self.self_attn = nn.Linear(hidden_size, hidden_size)
        self.mlp_gate = nn.Linear(hidden_size, intermediate_size)
        self.mlp_down = nn.Linear(intermediate_size, hidden_size)
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = torch.nn.functional.silu(self.mlp_gate(hidden_states))
        hidden_states = self.mlp_down(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class SimpleLlamaModel(nn.Module):
    """简化的 LLaMA 模型,用于测试"""

    def __init__(self, vocab_size=1000, hidden_size=128, num_layers=4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([SimpleLlamaLayer(hidden_size) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        hidden_states = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return type('Output', (), {'loss': loss, 'logits': logits})()


def create_test_batch(batch_size=2, seq_len=16, vocab_size=1000):
    """创建测试 batch"""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}


def get_model_gradients(model):
    """提取所有参数的梯度"""
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.clone()
    return grads


def compare_gradients(grads1, grads2, rtol=1e-5, atol=1e-6):
    """对比两组梯度,返回最大相对差异和差异最大的参数"""
    max_diff = 0.0
    max_diff_param = None
    diff_details = []

    all_params = set(grads1.keys()) | set(grads2.keys())

    for name in all_params:
        if name not in grads1:
            diff_details.append(f"{name}: missing in grads1")
            continue
        if name not in grads2:
            diff_details.append(f"{name}: missing in grads2")
            continue

        g1 = grads1[name]
        g2 = grads2[name]

        # 计算相对差异
        abs_diff = torch.abs(g1 - g2)
        rel_diff = abs_diff / (torch.abs(g1) + atol)
        max_rel_diff = rel_diff.max().item()

        if max_rel_diff > max_diff:
            max_diff = max_rel_diff
            max_diff_param = name

        if max_rel_diff > rtol:
            diff_details.append(
                f"{name}: max_rel_diff={max_rel_diff:.6e}, "
                f"abs_diff_mean={abs_diff.mean().item():.6e}"
            )

    return max_diff, max_diff_param, diff_details


def test_identity_codec_equals_baseline():
    """
    测试 identity codec 下,新实现与 baseline 数值完全等价
    """
    print(f"\n{'='*80}")
    print(f"Testing with synthetic SimpleLlamaModel")
    print(f"{'='*80}\n")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 创建测试 batch
    batch = create_test_batch(batch_size=2, seq_len=16, vocab_size=1000)
    batch = {k: v.to(device) for k, v in batch.items()}

    print(f"Batch shape: input_ids={batch['input_ids'].shape}")

    # ========================================================================
    # Baseline: 不启用 codec
    # ========================================================================
    print("\n" + "-" * 80)
    print("Running BASELINE (no codec)...")
    print("-" * 80)

    # 确保环境变量关闭
    os.environ["ENABLE_MIDDLE_LAYER_CODEC"] = "0"

    # 创建模型
    model_baseline = SimpleLlamaModel(vocab_size=1000, hidden_size=128, num_layers=4)
    model_baseline = model_baseline.to(device).to(torch.bfloat16)
    model_baseline.train()

    # Forward + backward
    outputs_baseline = model_baseline(**batch)
    loss_baseline = outputs_baseline.loss
    print(f"Baseline loss: {loss_baseline.item():.6f}")

    model_baseline.zero_grad()
    loss_baseline.backward()

    # 提取梯度
    grads_baseline = get_model_gradients(model_baseline)
    print(f"Baseline gradients: {len(grads_baseline)} parameters")

    # 清理
    del model_baseline
    torch.cuda.empty_cache()

    # ========================================================================
    # Test: 启用 identity codec
    # ========================================================================
    print("\n" + "-" * 80)
    print("Running TEST (identity codec)...")
    print("-" * 80)

    # 启用 codec,使用 identity
    os.environ["ENABLE_MIDDLE_LAYER_CODEC"] = "1"
    os.environ["CODEC_LAYER_IDX"] = "2"  # 4 层模型,选第 2 层
    os.environ["ACTIVATION_CODEC_TYPE"] = "identity"
    os.environ["GRADIENT_CODEC_TYPE"] = "identity"

    # 创建模型
    model_test = SimpleLlamaModel(vocab_size=1000, hidden_size=128, num_layers=4)
    model_test = model_test.to(device).to(torch.bfloat16)
    model_test.train()

    # 注入 codec hooks
    from llamafactory.train.finetune_hook import maybe_attach_middle_layer_hooks

    # 创建一个 dummy tokenizer 对象
    class DummyTokenizer:
        pad_token_id = 0

    cleanup_callback, codec_state, wrapped_collator = maybe_attach_middle_layer_hooks(
        model_test, DummyTokenizer(), None
    )

    assert cleanup_callback is not None, "Codec hooks should be installed"
    assert codec_state is not None, "CodecState should be created"

    print(f"Codec installed at layer {codec_state.layer_idx}")
    print(f"Activation codec: {codec_state.activation_codec.__class__.__name__}")
    print(f"Gradient codec: {codec_state.gradient_codec.__class__.__name__}")

    # Forward + backward
    outputs_test = model_test(**batch)
    loss_test = outputs_test.loss
    print(f"Test loss: {loss_test.item():.6f}")

    model_test.zero_grad()
    loss_test.backward()

    # 提取梯度
    grads_test = get_model_gradients(model_test)
    print(f"Test gradients: {len(grads_test)} parameters")

    # 清理
    cleanup_callback.on_train_end(None, None, None)
    del model_test
    torch.cuda.empty_cache()

    # ========================================================================
    # 对比结果
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    # 对比 loss
    loss_diff = abs(loss_baseline.item() - loss_test.item())
    print(f"\nLoss difference: {loss_diff:.6e}")

    # 对比梯度
    max_grad_diff, max_diff_param, diff_details = compare_gradients(
        grads_baseline, grads_test, rtol=1e-5, atol=1e-6
    )

    print(f"\nMax gradient relative diff: {max_grad_diff:.6e}")
    if max_diff_param:
        print(f"Max diff parameter: {max_diff_param}")

    if diff_details:
        print(f"\nParameters with diff > rtol:")
        for detail in diff_details[:10]:  # 只打印前 10 个
            print(f"  {detail}")
        if len(diff_details) > 10:
            print(f"  ... and {len(diff_details) - 10} more")

    # ========================================================================
    # 断言
    # ========================================================================
    print("\n" + "=" * 80)
    print("ASSERTIONS")
    print("=" * 80)

    # BF16 精度下,允许小的数值误差
    loss_threshold = 1e-4
    grad_threshold = 1e-3

    print(f"\nLoss threshold: {loss_threshold:.6e}")
    print(f"Gradient threshold: {grad_threshold:.6e}")

    assert loss_diff < loss_threshold, (
        f"Loss difference {loss_diff:.6e} exceeds threshold {loss_threshold:.6e}"
    )
    print("✅ Loss check passed")

    assert max_grad_diff < grad_threshold, (
        f"Max gradient diff {max_grad_diff:.6e} exceeds threshold {grad_threshold:.6e}"
    )
    print("✅ Gradient check passed")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    # 直接运行测试,不依赖 pytest
    test_identity_codec_equals_baseline()

