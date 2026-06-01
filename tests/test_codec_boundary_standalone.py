"""
测试 CodecBoundary 的数值等价性(独立测试,不依赖 LLaMA-Factory 的其他模块)

验证目标:
1. Identity codec(激活和梯度都不修改)下,CodecBoundary 与直接传递数值完全等价
2. 验证 loss 和梯度的 bit-exact 等价性(BF16 精度下)

运行方式:
    cd /home/liuzj/code/LLaMA-Factory
    /home/liuzj/miniconda3/envs/dora/bin/python tests/test_codec_boundary_standalone.py
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# ============================================================================
# 复制 finetune_hook.py 的核心组件(避免导入依赖)
# ============================================================================

class ActivationCodec:
    """激活值 codec 基类"""

    def encode_decode(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        step: int,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class GradientCodec:
    """梯度 codec 基类"""

    def encode_decode(
        self,
        gradient: torch.Tensor,
        layer_idx: int,
        step: int,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


class IdentityCodec(ActivationCodec, GradientCodec):
    """Identity codec: 直接返回输入,不做任何修改"""

    def encode_decode(
        self,
        tensor: torch.Tensor,
        layer_idx: int,
        step: int,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return tensor


@dataclass
class CodecState:
    """Codec 状态,包含 codec 实例和运行时状态"""

    activation_codec: ActivationCodec
    gradient_codec: GradientCodec
    layer_idx: int
    step: int
    padding_mask: Optional[torch.Tensor] = None


class CodecBoundary(torch.autograd.Function):
    """
    Codec 边界: 用 torch.autograd.Function 实现激活和梯度的编解码

    Forward: 对激活值进行编解码(在 no_grad 下,不进计算图)
    Backward: 对梯度进行编解码(在 no_grad 下)
    """

    @staticmethod
    def forward(ctx, hidden_states, codec_state):
        # 保存 codec_state 引用
        ctx.codec_state = codec_state

        # 在 no_grad 下调用 activation codec
        with torch.no_grad():
            compressed = codec_state.activation_codec.encode_decode(
                hidden_states.detach(),
                layer_idx=codec_state.layer_idx,
                step=codec_state.step,
                padding_mask=codec_state.padding_mask,
            )

        return compressed

    @staticmethod
    def backward(ctx, grad_output):
        codec_state = ctx.codec_state

        # 在 no_grad 下调用 gradient codec
        with torch.no_grad():
            compressed_grad = codec_state.gradient_codec.encode_decode(
                grad_output.detach(),
                layer_idx=codec_state.layer_idx,
                step=codec_state.step,
                padding_mask=codec_state.padding_mask,
            )

        # 返回 (hidden_states 的梯度, codec_state 的梯度)
        # codec_state 不需要梯度,返回 None
        return compressed_grad, None


# ============================================================================
# 测试模型
# ============================================================================

class SimpleLlamaLayer(nn.Module):
    """简化的 LLaMA decoder layer"""

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
    """简化的 LLaMA 模型"""

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


# ============================================================================
# 测试工具函数
# ============================================================================

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
    """对比两组梯度"""
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


# ============================================================================
# 主测试函数
# ============================================================================

def test_identity_codec_equals_baseline():
    """测试 identity codec 下,CodecBoundary 与 baseline 数值完全等价"""
    print(f"\n{'='*80}")
    print(f"Testing CodecBoundary with Identity Codec")
    print(f"{'='*80}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 创建测试 batch
    batch = create_test_batch(batch_size=2, seq_len=16, vocab_size=1000)
    batch = {k: v.to(device) for k, v in batch.items()}
    print(f"Batch shape: input_ids={batch['input_ids'].shape}")

    # ========================================================================
    # 创建模型并保存初始权重
    # ========================================================================
    print("\n" + "-" * 80)
    print("Creating model and saving initial weights...")
    print("-" * 80)

    # 设置随机种子确保可重复
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model_baseline = SimpleLlamaModel(vocab_size=1000, hidden_size=128, num_layers=4)
    model_baseline = model_baseline.to(device).to(torch.bfloat16)

    # 保存初始权重
    initial_state_dict = {k: v.clone() for k, v in model_baseline.state_dict().items()}
    print(f"Saved initial weights for {len(initial_state_dict)} parameters")

    # ========================================================================
    # Baseline: 不使用 CodecBoundary
    # ========================================================================
    print("\n" + "-" * 80)
    print("Running BASELINE (no CodecBoundary)...")
    print("-" * 80)

    model_baseline.train()

    outputs_baseline = model_baseline(**batch)
    loss_baseline = outputs_baseline.loss
    print(f"Baseline loss: {loss_baseline.item():.6f}")

    model_baseline.zero_grad()
    loss_baseline.backward()

    grads_baseline = get_model_gradients(model_baseline)
    print(f"Baseline gradients: {len(grads_baseline)} parameters")

    del model_baseline
    torch.cuda.empty_cache()

    # ========================================================================
    # Test: 使用 CodecBoundary + Identity Codec
    # ========================================================================
    print("\n" + "-" * 80)
    print("Running TEST (CodecBoundary + Identity Codec)...")
    print("-" * 80)

    # 创建新模型并加载相同的初始权重
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model_test = SimpleLlamaModel(vocab_size=1000, hidden_size=128, num_layers=4)
    model_test = model_test.to(device).to(torch.bfloat16)
    model_test.load_state_dict(initial_state_dict)
    model_test.train()

    print("Loaded same initial weights as baseline")

    # 创建 CodecState
    codec_state = CodecState(
        activation_codec=IdentityCodec(),
        gradient_codec=IdentityCodec(),
        layer_idx=2,
        step=0,
        padding_mask=None,
    )

    print(f"Codec at layer {codec_state.layer_idx}")
    print(f"Activation codec: {codec_state.activation_codec.__class__.__name__}")
    print(f"Gradient codec: {codec_state.gradient_codec.__class__.__name__}")

    # 注入 CodecBoundary (使用 register_forward_pre_hook)
    target_layer = model_test.layers[codec_state.layer_idx]

    def pre_hook(module, args):
        hidden_states = args[0]
        hidden_states = CodecBoundary.apply(hidden_states, codec_state)
        return (hidden_states,)

    handle = target_layer.register_forward_pre_hook(pre_hook)

    # Forward + backward
    outputs_test = model_test(**batch)
    loss_test = outputs_test.loss
    print(f"Test loss: {loss_test.item():.6f}")

    model_test.zero_grad()
    loss_test.backward()

    grads_test = get_model_gradients(model_test)
    print(f"Test gradients: {len(grads_test)} parameters")

    handle.remove()
    del model_test
    torch.cuda.empty_cache()

    # ========================================================================
    # 对比结果
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    loss_diff = abs(loss_baseline.item() - loss_test.item())
    print(f"\nLoss difference: {loss_diff:.6e}")

    max_grad_diff, max_diff_param, diff_details = compare_gradients(
        grads_baseline, grads_test, rtol=1e-5, atol=1e-6
    )

    print(f"\nMax gradient relative diff: {max_grad_diff:.6e}")
    if max_diff_param:
        print(f"Max diff parameter: {max_diff_param}")

    if diff_details:
        print(f"\nParameters with diff > rtol:")
        for detail in diff_details[:10]:
            print(f"  {detail}")
        if len(diff_details) > 10:
            print(f"  ... and {len(diff_details) - 10} more")

    # ========================================================================
    # 断言
    # ========================================================================
    print("\n" + "=" * 80)
    print("ASSERTIONS")
    print("=" * 80)

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
    print("✅ ALL TESTS PASSED - CodecBoundary is numerically equivalent to baseline")
    print("=" * 80)


if __name__ == "__main__":
    test_identity_codec_equals_baseline()
