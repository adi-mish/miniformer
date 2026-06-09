from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional

import torch

__all__ = [
    "GenerationConfig",
    "filter_logits_for_sampling",
    "sample_next_token",
]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


@dataclass(frozen=True)
class GenerationConfig:
    """Configuration for autoregressive decoding."""

    max_new_tokens: int = 32
    bos_token_id: int = 1
    eos_token_id: int = 2
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    use_cache: bool = True

    def validate(self, vocab_size: int) -> None:
        if not _is_int(vocab_size) or vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if not _is_int(self.max_new_tokens) or self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        for name, token_id in {
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }.items():
            if not _is_int(token_id) or token_id < 0 or token_id >= vocab_size:
                raise ValueError(f"{name} must be in [0, vocab_size)")
        if not isinstance(self.do_sample, bool):
            raise TypeError("do_sample must be a boolean")
        if not isinstance(self.use_cache, bool):
            raise TypeError("use_cache must be a boolean")
        if not _is_real(self.temperature) or not math.isfinite(float(self.temperature)):
            raise ValueError("temperature must be a finite number")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not _is_int(self.top_k) or self.top_k < 0 or self.top_k > vocab_size:
            raise ValueError("top_k must be in [0, vocab_size]")
        if not _is_real(self.top_p) or not math.isfinite(float(self.top_p)):
            raise ValueError("top_p must be a finite number")
        if self.top_p <= 0 or self.top_p > 1:
            raise ValueError("top_p must be in (0, 1]")
        if not self.do_sample and (self.temperature != 1.0 or self.top_k != 0 or self.top_p != 1.0):
            raise ValueError("temperature, top_k, and top_p require do_sample=True")


def _validate_logits(logits: torch.Tensor) -> None:
    if logits.dim() < 2:
        raise ValueError("logits must have shape [..., vocab_size]")
    if not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")


def _masked_fill_value(logits: torch.Tensor) -> float:
    return torch.finfo(logits.dtype).min


def filter_logits_for_sampling(
    logits: torch.Tensor,
    config: GenerationConfig,
) -> torch.Tensor:
    """Apply temperature, top-k, and nucleus filtering for sampling."""
    _validate_logits(logits)
    config.validate(logits.size(-1))
    if not config.do_sample:
        return logits

    filtered = logits / float(config.temperature)
    fill_value = _masked_fill_value(filtered)

    if config.top_k > 0:
        values, _ = torch.topk(filtered, config.top_k, dim=-1)
        min_keep = values[..., -1:]
        filtered = filtered.masked_fill(filtered < min_keep, fill_value)

    if config.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_mask = cumulative_probs > float(config.top_p)
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        remove_mask = torch.zeros_like(filtered, dtype=torch.bool)
        remove_mask.scatter_(dim=-1, index=sorted_idx, src=sorted_mask)
        filtered = filtered.masked_fill(remove_mask, fill_value)

    return filtered


def sample_next_token(
    logits: torch.Tensor,
    config: GenerationConfig,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Select the next token from logits using greedy or sampling semantics."""
    _validate_logits(logits)
    config.validate(logits.size(-1))
    if not config.do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    filtered = filter_logits_for_sampling(logits, config)
    probabilities = filtered.softmax(dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("sampling probabilities must contain only finite values")
    if (probabilities.sum(dim=-1) <= 0).any():
        raise ValueError("sampling probabilities must have non-zero mass")

    vocab_size = probabilities.size(-1)
    flat_probabilities = probabilities.reshape(-1, vocab_size)
    sampled = torch.multinomial(flat_probabilities, num_samples=1, generator=generator)
    return sampled.reshape(*probabilities.shape[:-1], 1)
