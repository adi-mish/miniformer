"""Small deterministic tokenization helpers."""

from __future__ import annotations

import hashlib


def stable_token_id(token: object, vocab_size: int) -> int:
    """Map a token-like value to a reproducible integer id."""
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    digest = hashlib.blake2b(str(token).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % vocab_size
