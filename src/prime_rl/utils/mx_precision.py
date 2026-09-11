"""Translate a model's precision requirements into ModelExpress tensor overrides."""

import torch
from torch import nn


def wire_dtype_overrides(model: nn.Module, tensors: dict[str, torch.Tensor]) -> dict[str, torch.dtype]:
    keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
    if keep_in_fp32 is None:
        return {}
    return {name: torch.float32 for name in tensors if keep_in_fp32(name)}
