import pytest
import torch

from miniformer.train.pooling import pool_sequence_outputs


def test_first_pooling_selects_first_position():
    outputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    pooled = pool_sequence_outputs(outputs, None, mode="first")

    assert torch.equal(pooled, torch.tensor([[1.0, 2.0]]))


def test_mean_pooling_uses_all_positions():
    outputs = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])

    pooled = pool_sequence_outputs(outputs, None, mode="mean")

    assert torch.equal(pooled, torch.tensor([[3.0, 5.0]]))


def test_masked_mean_excludes_padding():
    outputs = torch.tensor(
        [
            [[1.0, 3.0], [5.0, 7.0], [100.0, 100.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, True, True]])

    pooled = pool_sequence_outputs(outputs, mask, mode="masked_mean")

    assert torch.equal(pooled[0], torch.tensor([3.0, 5.0]))
    assert torch.equal(pooled[1], torch.tensor([6.0, 8.0]))


def test_masked_mean_requires_matching_mask():
    with pytest.raises(ValueError, match="attention_mask shape"):
        pool_sequence_outputs(
            torch.zeros(2, 3, 4),
            torch.ones(2, 2, dtype=torch.bool),
            mode="masked_mean",
        )
