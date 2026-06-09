from __future__ import annotations

from typing import Optional

import torch


def padding_mask(seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """Return a boolean key-padding mask with shape [batch, 1, 1, seq_len]."""
    if seq.dim() < 2:
        raise ValueError("seq must have at least [batch, seq_len] dimensions")

    batch_size, seq_len = seq.size(0), seq.size(1)
    if seq.dim() == 2 and seq.dtype == torch.long:
        return (seq != pad_id).unsqueeze(1).unsqueeze(2)
    return torch.ones(
        batch_size,
        1,
        1,
        seq_len,
        device=seq.device,
        dtype=torch.bool,
    )


def causal_mask(
    query_len: int,
    *,
    key_len: Optional[int] = None,
    past_len: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a cache-aware causal mask with shape [1, 1, query_len, key_len]."""
    if query_len <= 0:
        raise ValueError("query_len must be positive")
    if past_len < 0:
        raise ValueError("past_len must be non-negative")
    if key_len is None:
        key_len = past_len + query_len
    if key_len <= 0:
        raise ValueError("key_len must be positive")
    if key_len < past_len + query_len:
        raise ValueError("key_len must cover past_len + query_len")

    query_positions = torch.arange(
        past_len,
        past_len + query_len,
        device=device,
    ).unsqueeze(1)
    key_positions = torch.arange(key_len, device=device).unsqueeze(0)
    return (key_positions <= query_positions).unsqueeze(0).unsqueeze(0)


def validate_attention_mask(
    mask: torch.Tensor,
    *,
    query_len: Optional[int] = None,
    key_len: Optional[int] = None,
    batch_size: Optional[int] = None,
    name: str = "attention mask",
) -> torch.Tensor:
    """Validate that a boolean mask can broadcast over [batch, heads, query, key]."""
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must be a boolean tensor")
    if mask.dim() < 2 or mask.dim() > 4:
        raise ValueError(f"{name} must have 2 to 4 dimensions")

    if query_len is not None and mask.size(-2) not in {1, query_len}:
        raise ValueError(f"{name} query dimension must be 1 or {query_len}, got {mask.size(-2)}")
    if key_len is not None and mask.size(-1) not in {1, key_len}:
        raise ValueError(f"{name} key dimension must be 1 or {key_len}, got {mask.size(-1)}")
    if batch_size is not None and mask.dim() >= 3 and mask.size(0) not in {1, batch_size}:
        raise ValueError(f"{name} batch dimension must be 1 or {batch_size}, got {mask.size(0)}")

    return mask


def combine_masks(*masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Combine broadcastable boolean masks with logical AND."""
    combined: Optional[torch.Tensor] = None
    for mask in masks:
        if mask is None:
            continue
        validate_attention_mask(mask)
        try:
            combined = mask if combined is None else combined & mask
        except RuntimeError as exc:
            raise ValueError("attention masks are not broadcastable") from exc
    return combined


def self_attention_mask(
    seq: torch.Tensor,
    *,
    causal: bool,
    pad_id: int = 0,
) -> torch.Tensor:
    """Build the canonical self-attention mask for token or feature inputs."""
    pad = padding_mask(seq, pad_id=pad_id)
    if not causal:
        return pad
    combined = combine_masks(pad, causal_mask(seq.size(1), device=seq.device))
    assert combined is not None
    return combined
