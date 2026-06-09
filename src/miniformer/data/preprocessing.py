from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import torch

from miniformer.utils.tokenization import stable_token_id


class TextTokenizer(Protocol):
    """Minimal tokenizer interface used by data preprocessing."""

    def encode(self, text: str, add_special_tokens: bool = True) -> Sequence[int]:
        """Encode text into integer token IDs."""
        ...


def encode_text(
    text: object,
    *,
    vocab_size: int | None = None,
    tokenizer: TextTokenizer | None = None,
) -> torch.Tensor:
    """Encode one text value into a 1D long tensor."""
    if tokenizer is not None:
        ids = list(tokenizer.encode(str(text), add_special_tokens=True))
    else:
        if vocab_size is None:
            raise ValueError("vocab_size is required when tokenizer is not provided")
        ids = [stable_token_id(token, vocab_size) for token in str(text).split()]
    return torch.tensor(ids, dtype=torch.long)


def pad_token_sequences(
    sequences: Sequence[torch.Tensor],
    *,
    pad_id: int = 0,
) -> torch.Tensor:
    """Pad a non-empty sequence of 1D token tensors to [batch, max_len]."""
    if not sequences:
        raise ValueError("sequences must not be empty")
    if any(sequence.dim() != 1 for sequence in sequences):
        raise ValueError("token sequences must be 1D tensors")
    if any(sequence.numel() == 0 for sequence in sequences):
        raise ValueError("String inputs must not be empty")

    max_len = max(sequence.size(0) for sequence in sequences)
    padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        padded[index, : sequence.size(0)] = sequence
    return padded


def encode_text_batch(
    texts: Sequence[object],
    *,
    vocab_size: int,
    tokenizer: TextTokenizer | None = None,
    pad_id: int = 0,
) -> torch.Tensor:
    """Encode and pad text values into an integer token batch."""
    return pad_token_sequences(
        [encode_text(text, vocab_size=vocab_size, tokenizer=tokenizer) for text in texts],
        pad_id=pad_id,
    )


def _collate_language_modeling(batch: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    for record in batch:
        input_ids = record.get("input_ids")
        labels = record.get("labels")
        if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise ValueError("language_modeling batches require tensor input_ids and labels")
        if input_ids.dim() != 1 or labels.dim() != 1:
            raise ValueError("language_modeling input_ids and labels must be 1D tensors")
        if input_ids.size(0) != labels.size(0):
            raise ValueError("language_modeling input_ids and labels must have matching lengths")
        if input_ids.numel() == 0:
            raise ValueError("language_modeling records must not be empty")

    lengths = [record["input_ids"].size(0) for record in batch]
    max_len = max(lengths)
    input_ids = torch.full((len(batch), max_len), 0, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    for index, record in enumerate(batch):
        seq_len = lengths[index]
        input_ids[index, :seq_len] = record["input_ids"]
        labels[index, :seq_len] = record["labels"]
    return {"input_ids": input_ids, "labels": labels}


def _collate_text_supervision(
    batch: Sequence[Mapping[str, Any]],
    *,
    task: str,
    vocab_size: int,
    tokenizer: TextTokenizer | None,
) -> dict[str, torch.Tensor]:
    label_dtype = torch.long if task == "classification" else torch.float32
    return {
        "input_ids": encode_text_batch(
            [record["input"] for record in batch],
            vocab_size=vocab_size,
            tokenizer=tokenizer,
        ),
        "labels": torch.as_tensor(
            [_supervised_label(record, task=task) for record in batch],
            dtype=label_dtype,
        ),
    }


def _collate_numeric_features(
    batch: Sequence[Mapping[str, Any]],
    *,
    task: str,
) -> dict[str, torch.Tensor]:
    raw_sequences = [record["input"] for record in batch]
    for sequence in raw_sequences:
        if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes)):
            raise ValueError("Numeric feature inputs must be sequences of mapping steps")
        if len(sequence) == 0:
            raise ValueError("Numeric feature sequences must not be empty")
        if any(not isinstance(step, Mapping) for step in sequence):
            raise ValueError("Numeric feature steps must be mappings")

    seq_lens = [len(sequence) for sequence in raw_sequences]
    if min(seq_lens) == 0:
        raise ValueError("Numeric feature sequences must not be empty")
    max_len = max(seq_lens)
    first_sequence = raw_sequences[0]
    feat_keys = list(first_sequence[0].keys())
    feat_dim = len(feat_keys)
    if feat_dim == 0:
        raise ValueError("Numeric feature steps must contain at least one feature")

    inputs = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    label_dtype = torch.long if task == "classification" else torch.float32
    labels = torch.as_tensor(
        [_supervised_label(record, task=task) for record in batch],
        dtype=label_dtype,
    )
    for index, sequence in enumerate(raw_sequences):
        for step in sequence:
            if set(step.keys()) != set(feat_keys):
                raise ValueError("Numeric feature steps must share the same feature keys")
        tensor = torch.tensor(
            [[step[key] for key in feat_keys] for step in sequence],
            dtype=torch.float32,
        )
        inputs[index, : tensor.size(0)] = tensor
    return {"input": inputs, "labels": labels}


def _supervised_label(record: Mapping[str, Any], *, task: str) -> Any:
    if "labels" in record:
        return _as_scalar_label(record["labels"])
    if "label" in record:
        return _as_scalar_label(record["label"])
    if task == "regression" and "value" in record:
        return _as_scalar_label(record["value"])
    accepted = "label, labels, or value" if task == "regression" else "label or labels"
    raise ValueError(f"{task} records must include a {accepted} field")


def _as_scalar_label(label: Any) -> Any:
    if isinstance(label, torch.Tensor):
        if label.numel() != 1:
            raise ValueError("Supervised labels must be scalar values")
        return label.item()
    return label


def collate_records(
    batch: Sequence[Mapping[str, Any]],
    *,
    task: str,
    vocab_size: int,
    tokenizer: TextTokenizer | None = None,
) -> dict[str, torch.Tensor]:
    """Collate JSONL records into tensor batches for the training module."""
    if not batch:
        raise ValueError("batch must not be empty")
    if any(not isinstance(record, Mapping) for record in batch):
        raise ValueError("batch records must be mappings")

    if task == "language_modeling":
        return _collate_language_modeling(batch)

    if task in {"classification", "regression"}:
        first_input = batch[0].get("input")
        if isinstance(first_input, str):
            return _collate_text_supervision(
                batch,
                task=task,
                vocab_size=vocab_size,
                tokenizer=tokenizer,
            )
        return _collate_numeric_features(batch, task=task)

    raise ValueError(f"Unsupported task for collation: {task}")
