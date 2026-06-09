from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from miniformer.utils.tokenization import stable_token_id


@runtime_checkable
class TokenizerProtocol(Protocol):
    """Minimal tokenizer interface used by Miniformer data preprocessing."""

    def encode(self, text: str, add_special_tokens: bool = True) -> Sequence[int]:
        """Encode text into integer token IDs."""
        ...


TextTokenizer = TokenizerProtocol


@dataclass(frozen=True)
class WhitespaceHashTokenizer:
    """Deterministic whitespace tokenizer backed by stable token hashing."""

    vocab_size: int
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        if self.vocab_size <= max(self.bos_token_id, self.eos_token_id):
            raise ValueError("vocab_size must be larger than special token IDs")
        if min(self.bos_token_id, self.eos_token_id) < 0:
            raise ValueError("special token IDs must be non-negative")

    def encode(self, text: str, add_special_tokens: bool = True) -> Sequence[int]:
        token_ids = [stable_token_id(token, self.vocab_size) for token in str(text).split()]
        if add_special_tokens:
            return [self.bos_token_id, *token_ids, self.eos_token_id]
        return token_ids


def ensure_tokenizer(
    tokenizer: TokenizerProtocol | None,
    *,
    vocab_size: int,
) -> TokenizerProtocol:
    """Return an explicit tokenizer, falling back to WhitespaceHashTokenizer."""
    if tokenizer is None:
        return WhitespaceHashTokenizer(vocab_size=vocab_size)
    if not isinstance(tokenizer, TokenizerProtocol):
        raise TypeError("tokenizer must implement encode(text, add_special_tokens=True)")
    return tokenizer
