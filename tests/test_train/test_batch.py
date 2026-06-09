import pytest
import torch

from miniformer.data.batch import Batch


def test_batch_from_mapping_accepts_token_dict():
    batch = Batch.from_mapping(
        {
            "input_ids": torch.tensor([[1, 2, 0]]),
            "attention_mask": torch.tensor([[True, True, False]]),
            "labels": torch.tensor([1]),
        }
    )

    assert batch.kind == "tokens"
    assert batch.inputs.shape == (1, 3)
    assert batch.attention_mask is not None
    assert batch.to_dict()["input_ids"].shape == (1, 3)


def test_batch_from_mapping_accepts_feature_dict():
    batch = Batch.from_mapping(
        {
            "input": torch.randn(2, 3, 4),
            "attention_mask": torch.ones(2, 3, dtype=torch.bool),
            "labels": torch.tensor([0, 1]),
        }
    )

    assert batch.kind == "features"
    assert "input" in batch.to_dict()


def test_batch_rejects_raw_text_mapping():
    with pytest.raises(TypeError, match="Raw text"):
        Batch.from_mapping({"input": "hello", "labels": torch.tensor([0])})


def test_batch_rejects_bad_mask_shape():
    with pytest.raises(ValueError, match="attention_mask"):
        Batch(
            inputs=torch.ones(2, 3, dtype=torch.long),
            attention_mask=torch.ones(2, 2, dtype=torch.bool),
        )
