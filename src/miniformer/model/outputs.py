from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from miniformer.model.decoder import AttentionList, DecoderPastKeyValues


@dataclass(frozen=True)
class TransformerModelOutput:
    """Explicit encoder-only transformer output."""

    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    projection: Optional[torch.Tensor] = None
    self_attentions: Optional[AttentionList] = None
    past_key_values: Optional[Any] = None

    @property
    def output(self) -> torch.Tensor:
        if self.logits is not None:
            return self.logits
        if self.projection is not None:
            return self.projection
        if self.hidden_states is not None:
            return self.hidden_states
        raise RuntimeError("TransformerModelOutput has no tensor output")


@dataclass(frozen=True)
class Seq2SeqModelOutput:
    """Explicit encoder-decoder transformer output."""

    logits: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    projection: Optional[torch.Tensor] = None
    self_attentions: Optional[AttentionList] = None
    cross_attentions: Optional[AttentionList] = None
    past_key_values: Optional[DecoderPastKeyValues] = None

    @property
    def output(self) -> torch.Tensor:
        if self.logits is not None:
            return self.logits
        if self.projection is not None:
            return self.projection
        if self.hidden_states is not None:
            return self.hidden_states
        raise RuntimeError("Seq2SeqModelOutput has no tensor output")
