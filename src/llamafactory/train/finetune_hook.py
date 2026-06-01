import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import TrainerCallback
from typing_extensions import override

from ..extras.logging import get_logger


logger = get_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


# ============================================================================
# Codec Interface
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
        grad_output: torch.Tensor,
        layer_idx: int,
        step: int,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


# ============================================================================
# CodecState
# ============================================================================


@dataclass
class CodecState:
    """管理 codec 实例、step 计数器、padding_mask"""

    activation_codec: ActivationCodec
    gradient_codec: GradientCodec
    layer_idx: int
    step: int = 0
    padding_mask: Optional[torch.Tensor] = None


# ============================================================================
# CodecBoundary (torch.autograd.Function)
# ============================================================================


class CodecBoundary(torch.autograd.Function):
    """
    使用 torch.autograd.Function 实现 codec 边界。
    - forward: 调用 activation_codec.encode_decode (在 no_grad 下)
    - backward: 调用 gradient_codec.encode_decode (在 no_grad 下)
    """

    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, codec_state: CodecState) -> torch.Tensor:
        ctx.codec_state = codec_state

        with torch.no_grad():
            h_compressed = codec_state.activation_codec.encode_decode(
                hidden_states.detach(),
                layer_idx=codec_state.layer_idx,
                step=codec_state.step,
                padding_mask=codec_state.padding_mask,
            )

        return h_compressed

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        codec_state = ctx.codec_state

        with torch.no_grad():
            grad_compressed = codec_state.gradient_codec.encode_decode(
                grad_output.detach(),
                layer_idx=codec_state.layer_idx,
                step=codec_state.step,
                padding_mask=codec_state.padding_mask,
            )

        return grad_compressed, None


# ============================================================================
# Concrete Codec Implementations
# ============================================================================


class IdentityCodec(ActivationCodec, GradientCodec):
    """Identity codec: 直接返回输入，用于验证"""

    def encode_decode(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        step: int,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return hidden_states


class UniformQuantizationCodec(ActivationCodec):
    """均匀量化 codec, 按 bit 数参数化 (2/4/8 bit)。

    量化范围按最后一维 (channel) 的 min/max 动态确定。
    """

    def __init__(self, bits: int):
        if bits < 1:
            raise ValueError(f"bits must be >= 1, got {bits}")
        self.bits = bits
        self.levels = (1 << bits) - 1  # 2bit->3, 4bit->15, 8bit->255

    def encode_decode(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        step: int,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = hidden_states
        min_val = h.min(dim=-1, keepdim=True).values
        max_val = h.max(dim=-1, keepdim=True).values
        scale = ((max_val - min_val) / self.levels).clamp(min=1e-8)

        normalized = (h - min_val) / scale
        quantized = normalized.round().clamp(0, self.levels)
        h_out = quantized * scale + min_val

        return h_out


# ============================================================================
# Codec Factory Functions
# ============================================================================


def _get_activation_codec() -> ActivationCodec:
    """根据环境变量返回对应的 activation codec"""
    codec_type = os.getenv("ACTIVATION_CODEC", "identity").lower()

    if codec_type == "identity":
        return IdentityCodec()
    elif codec_type == "uniform_2bit":
        return UniformQuantizationCodec(bits=2)
    elif codec_type == "uniform_4bit":
        return UniformQuantizationCodec(bits=4)
    elif codec_type == "uniform_8bit":
        return UniformQuantizationCodec(bits=8)
    else:
        raise ValueError(f"Unknown activation codec type: {codec_type}")


def _get_gradient_codec() -> GradientCodec:
    """根据环境变量返回对应的 gradient codec (默认 identity)"""
    codec_type = os.getenv("GRADIENT_CODEC", "identity").lower()

    if codec_type == "identity":
        return IdentityCodec()
    else:
        raise ValueError(f"Unknown gradient codec type: {codec_type}")


# ============================================================================
# DataCollator with Padding Mask
# ============================================================================


class DataCollatorWithPaddingMask:
    """
    包装基础 collator, 在 __call__ 里构造 padding_mask。
    padding_mask 形状: (B, T), True 表示 padding 位置。
    """

    def __init__(self, base_collator: Any, pad_token_id: int):
        self.base_collator = base_collator
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.base_collator(features)

        # 构造 padding_mask: (B, T), True 表示 padding
        if "input_ids" in batch:
            input_ids = batch["input_ids"]
            padding_mask = input_ids == self.pad_token_id
            batch["padding_mask"] = padding_mask

        return batch


# ============================================================================
# CodecStateInjectorCallback
# ============================================================================


class CodecStateInjectorCallback(TrainerCallback):
    """
    在 on_step_begin 里从 inputs pop("padding_mask"), 注入到 codec_state.padding_mask。
    在 on_step_end 里清空 codec_state.padding_mask。
    """

    def __init__(self, codec_state: CodecState):
        self.codec_state = codec_state

    @override
    def on_step_begin(self, args, state, control, **kwargs):
        # 从 inputs 中提取 padding_mask
        if "inputs" in kwargs:
            inputs = kwargs["inputs"]
            if isinstance(inputs, dict) and "padding_mask" in inputs:
                padding_mask = inputs.pop("padding_mask")
                self.codec_state.padding_mask = padding_mask

        # 更新 step 计数器
        self.codec_state.step = state.global_step

    @override
    def on_step_end(self, args, state, control, **kwargs):
        # 清空 padding_mask
        self.codec_state.padding_mask = None


# ============================================================================
# HookCleanupCallback
# ============================================================================


class HookCleanupCallback(TrainerCallback):
    """在训练结束时清理 hook"""

    def __init__(self, cleanup_fn: Callable[[], None]):
        self.cleanup_fn = cleanup_fn

    @override
    def on_train_end(self, args, state, control, **kwargs):
        logger.info("Cleaning up middle layer hooks...")
        self.cleanup_fn()


# ============================================================================
# Helper Functions
# ============================================================================


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """递归解包 DDP/FSDP/DeepSpeed/PEFT 等包装器"""
    # 处理 DDP/FSDP/DeepSpeed 的 module 属性
    if hasattr(model, "module"):
        return _unwrap_model(model.module)
    # 处理 PEFT 的 base_model 属性
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def _find_layer_list(model: torch.nn.Module) -> Optional[torch.nn.ModuleList]:
    """
    查找模型的 layer list。
    支持 LLaMA/Qwen/Mistral 等常见架构。
    """
    unwrapped = _unwrap_model(model)

    # 尝试常见的路径
    candidates = [
        "model.layers",  # LLaMA, Qwen, Mistral
        "transformer.h",  # GPT-2
        "transformer.layers",  # GPT-J
        "encoder.layer",  # BERT
    ]

    for path in candidates:
        obj = unwrapped
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                obj = None
                break
        if obj is not None and isinstance(obj, torch.nn.ModuleList):
            return obj

    return None


# ============================================================================
# Main Hook Installation Function
# ============================================================================


def maybe_attach_middle_layer_hooks(
    model: torch.nn.Module,
    tokenizer: Any,
    data_collator: Any,
) -> Tuple[Optional[HookCleanupCallback], Optional[CodecState], Any]:
    """
    在指定层注入 CodecBoundary。
    返回: (cleanup_callback, codec_state, wrapped_data_collator)
    """
    # 检查是否启用
    if not _env_flag("ENABLE_MIDDLE_LAYER_CODEC", default=False):
        logger.info("Middle layer codec disabled (ENABLE_MIDDLE_LAYER_CODEC not set)")
        return None, None, data_collator

    # 获取目标层索引
    target_layer_idx = _env_int("CODEC_LAYER_IDX", default=16)
    logger.info(f"Installing codec boundary at layer {target_layer_idx}")

    # 查找 layer list
    layer_list = _find_layer_list(model)
    if layer_list is None:
        raise RuntimeError("Could not find model.layers (or equivalent) in the model")

    if target_layer_idx < 0 or target_layer_idx >= len(layer_list):
        raise ValueError(
            f"Invalid CODEC_LAYER_IDX={target_layer_idx}, model has {len(layer_list)} layers"
        )

    # 创建 codec 实例
    activation_codec = _get_activation_codec()
    gradient_codec = _get_gradient_codec()
    logger.info(f"Activation codec: {activation_codec.__class__.__name__}")
    logger.info(f"Gradient codec: {gradient_codec.__class__.__name__}")

    # 创建 CodecState
    codec_state = CodecState(
        activation_codec=activation_codec,
        gradient_codec=gradient_codec,
        layer_idx=target_layer_idx,
        step=0,
        padding_mask=None,
    )

    # 注入 CodecBoundary (使用 register_forward_pre_hook)
    target_layer = layer_list[target_layer_idx]
    hook_handles = []

    def pre_hook(module, args):
        hidden_states = args[0]
        # 在该层 forward 之前插入 CodecBoundary
        hidden_states = CodecBoundary.apply(hidden_states, codec_state)
        return (hidden_states,) + args[1:]

    handle = target_layer.register_forward_pre_hook(pre_hook)
    hook_handles.append(handle)

    # 定义 cleanup 函数
    def cleanup():
        for h in hook_handles:
            h.remove()
        logger.info("Removed codec boundary hooks")

    cleanup_callback = HookCleanupCallback(cleanup)

    # 包装 data_collator
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    wrapped_collator = DataCollatorWithPaddingMask(data_collator, pad_token_id)

    logger.info("Codec boundary installed successfully")
    return cleanup_callback, codec_state, wrapped_collator

