import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

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
