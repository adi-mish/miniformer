from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

BatchKind = Literal["tokens", "features", "language_modeling"]


@dataclass(frozen=True)
class Batch:
    """Typed tensor batch used internally by training code."""

    inputs: torch.Tensor
    labels: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    kind: BatchKind = "tokens"

    def __post_init__(self) -> None:
        if self.inputs.dim() < 2:
            raise ValueError("Batch inputs must have at least [batch, seq] dimensions")
        if self.kind not in {"tokens", "features", "language_modeling"}:
            raise ValueError(f"Unknown batch kind: {self.kind}")
        if self.attention_mask is not None:
            if self.attention_mask.dtype != torch.bool:
                raise TypeError("Batch attention_mask must be a boolean tensor")
            if self.attention_mask.shape != self.inputs.shape[:2]:
                raise ValueError("Batch attention_mask must match input batch and sequence dims")
        if self.labels is not None and self.labels.size(0) != self.inputs.size(0):
            raise ValueError("Batch labels must have the same batch size as inputs")

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            inputs=self.inputs.to(device),
            labels=self.labels.to(device) if self.labels is not None else None,
            attention_mask=(
                self.attention_mask.to(device, dtype=torch.bool)
                if self.attention_mask is not None
                else None
            ),
            kind=self.kind,
        )

    def with_attention_mask(self, attention_mask: torch.Tensor) -> "Batch":
        return Batch(
            inputs=self.inputs,
            labels=self.labels,
            attention_mask=attention_mask,
            kind=self.kind,
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        kind: BatchKind | None = None,
    ) -> "Batch":
        if isinstance(mapping.get("input_ids"), torch.Tensor):
            inputs = mapping["input_ids"]
            inferred_kind: BatchKind = "tokens"
        elif isinstance(mapping.get("input"), torch.Tensor):
            inputs = mapping["input"]
            inferred_kind = "features"
        else:
            raise TypeError(
                "Batch mapping requires tensor input_ids or tensor input. "
                "Raw text and records belong in miniformer.data.preprocessing."
            )

        labels = mapping.get("labels")
        if labels is not None and not isinstance(labels, torch.Tensor):
            raise TypeError("Batch labels must be a tensor when present")

        attention_mask = mapping.get("attention_mask")
        if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            raise TypeError("Batch attention_mask must be a tensor when present")

        return cls(
            inputs=inputs,
            labels=labels,
            attention_mask=attention_mask,
            kind=kind or inferred_kind,
        )

    def to_dict(self) -> dict[str, torch.Tensor]:
        key = "input" if self.kind == "features" else "input_ids"
        output = {key: self.inputs}
        if self.labels is not None:
            output["labels"] = self.labels
        if self.attention_mask is not None:
            output["attention_mask"] = self.attention_mask
        return output
