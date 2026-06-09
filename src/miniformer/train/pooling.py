from __future__ import annotations

from typing import Literal

import torch

PoolingMode = Literal["first", "mean", "masked_mean"]


def pool_sequence_outputs(
    outputs: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    mode: PoolingMode,
) -> torch.Tensor:
    """Pool [batch, seq, dim] supervised model outputs into [batch, dim]."""
    if outputs.dim() != 3:
        return outputs

    if mode == "first":
        return outputs[:, 0, :]
    if mode == "mean":
        return outputs.mean(dim=1)
    if mode != "masked_mean":
        raise ValueError(f"Unknown pooling mode: {mode}")
    if attention_mask is None:
        raise ValueError("masked_mean pooling requires an attention_mask")
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must have shape [batch, seq_len]")
    if attention_mask.shape != outputs.shape[:2]:
        raise ValueError("attention_mask shape must match output batch and sequence dimensions")

    weights = attention_mask.to(device=outputs.device, dtype=outputs.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (outputs * weights).sum(dim=1) / denominator
