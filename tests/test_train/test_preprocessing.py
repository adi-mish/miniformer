import pytest
import torch

from miniformer.data.preprocessing import (
    attention_mask_from_lengths,
    collate_records,
    encode_text,
    encode_text_batch,
    pad_token_sequences,
)


class DummyTokenizer:
    def encode(self, text, add_special_tokens=True):
        values = [ord(char) % 100 for char in text]
        return [1, *values, 2] if add_special_tokens else values


def test_encode_text_requires_vocab_without_tokenizer():
    with pytest.raises(ValueError, match="vocab_size"):
        encode_text("hello")


def test_encode_text_uses_tokenizer_when_supplied():
    ids = encode_text("ab", tokenizer=DummyTokenizer())

    assert torch.equal(ids, torch.tensor([1, 97, 98, 2], dtype=torch.long))


def test_encode_text_batch_hashes_and_pads_text():
    batch = encode_text_batch(["hello world", "hello"], vocab_size=100)

    assert batch.shape == (2, 2)
    assert batch.dtype == torch.long
    assert batch[0, 0] == batch[1, 0]
    assert batch[1, 1] == 0


def test_pad_token_sequences_rejects_empty_sequences():
    with pytest.raises(ValueError, match="must not be empty"):
        pad_token_sequences([torch.tensor([], dtype=torch.long)])


def test_attention_mask_from_lengths():
    mask = attention_mask_from_lengths([2, 1], max_len=3)

    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, True, False], [True, False, False]]


def test_collate_records_string_classification_returns_tensor_batch():
    batch = collate_records(
        [{"input": "alpha beta", "label": 1}, {"input": "alpha", "labels": 0}],
        task="classification",
        vocab_size=50,
    )

    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert batch["input_ids"].shape == (2, 2)
    assert batch["attention_mask"].tolist() == [[True, True], [True, False]]
    assert batch["labels"].dtype == torch.long
    assert batch["labels"].tolist() == [1, 0]


def test_collate_records_string_regression_accepts_value_field():
    batch = collate_records(
        [{"input": "alpha", "value": 1.5}, {"input": "beta", "labels": 2.5}],
        task="regression",
        vocab_size=50,
    )

    assert batch["input_ids"].shape == (2, 1)
    assert batch["attention_mask"].tolist() == [[True], [True]]
    assert batch["labels"].dtype == torch.float32
    assert batch["labels"].tolist() == pytest.approx([1.5, 2.5])


def test_collate_records_numeric_features_are_padded_and_typed():
    batch = collate_records(
        [
            {"input": [{"a": 1.0, "b": 2.0}], "labels": 1},
            {"input": [{"a": 3.0, "b": 4.0}, {"a": 5.0, "b": 6.0}], "labels": 0},
        ],
        task="classification",
        vocab_size=50,
    )

    assert torch.equal(batch["input"][0, 0], torch.tensor([1.0, 2.0]))
    assert torch.equal(batch["input"][0, 1], torch.tensor([0.0, 0.0]))
    assert batch["attention_mask"].tolist() == [[True, False], [True, True]]
    assert batch["labels"].tolist() == [1, 0]


def test_collate_records_numeric_features_reject_mismatched_keys():
    with pytest.raises(ValueError, match="same feature keys"):
        collate_records(
            [{"input": [{"a": 1.0}, {"b": 2.0}], "labels": 0}],
            task="classification",
            vocab_size=50,
        )


def test_collate_records_language_modeling_pads_labels_with_ignore_index():
    batch = collate_records(
        [
            {"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([2, 3])},
            {"input_ids": torch.tensor([4]), "labels": torch.tensor([5])},
        ],
        task="language_modeling",
        vocab_size=50,
    )

    assert batch["input_ids"].tolist() == [[1, 2], [4, 0]]
    assert batch["attention_mask"].tolist() == [[True, True], [True, False]]
    assert batch["labels"].tolist() == [[2, 3], [5, -100]]
