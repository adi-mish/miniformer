from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass(frozen=True)
class KeyValueCache:
    """Projected key/value tensors for one attention module."""

    key: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class DecoderLayerCache:
    """Self-attention and cross-attention caches for one decoder layer."""

    self_attention: Optional[KeyValueCache] = None
    cross_attention: Optional[KeyValueCache] = None


AttentionList = List[Optional[torch.Tensor]]
EncoderPastKeyValues = List[Optional[KeyValueCache]]
DecoderPastKeyValues = List[DecoderLayerCache]
