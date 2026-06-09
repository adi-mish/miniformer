import pytest
import torch

from miniformer.model.masks import (
    causal_mask,
    combine_masks,
    padding_mask,
    self_attention_mask,
    validate_attention_mask,
)


def test_padding_mask_marks_token_padding_keys_only():
    tokens = torch.tensor([[4, 0, 7], [0, 2, 3]])

    mask = padding_mask(tokens)

    assert mask.shape == (2, 1, 1, 3)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[[[True, False, True]]], [[[False, True, True]]]]


def test_padding_mask_for_feature_inputs_is_all_visible():
    features = torch.randn(2, 5, 3)

    mask = padding_mask(features)

    assert mask.shape == (2, 1, 1, 5)
    assert mask.all()


def test_causal_mask_is_cache_aware():
    mask = causal_mask(query_len=2, past_len=3)

    assert mask.shape == (1, 1, 2, 5)
    assert mask[0, 0].tolist() == [
        [True, True, True, True, False],
        [True, True, True, True, True],
    ]


def test_causal_mask_rejects_invalid_lengths():
    with pytest.raises(ValueError, match="query_len"):
        causal_mask(0)
    with pytest.raises(ValueError, match="past_len"):
        causal_mask(1, past_len=-1)
    with pytest.raises(ValueError, match="key_len"):
        causal_mask(3, key_len=2)


def test_combine_masks_broadcasts_padding_and_causal_masks():
    tokens = torch.tensor([[5, 6, 0]])

    mask = combine_masks(padding_mask(tokens), causal_mask(3, device=tokens.device))

    assert mask is not None
    assert mask.shape == (1, 1, 3, 3)
    assert mask[0, 0].tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, False],
    ]


def test_combine_masks_rejects_unbroadcastable_shapes():
    with pytest.raises(ValueError, match="broadcastable"):
        combine_masks(
            torch.ones(1, 1, 2, 3, dtype=torch.bool),
            torch.ones(1, 1, 4, 3, dtype=torch.bool),
        )


def test_validate_attention_mask_rejects_bad_dtype_and_dimensions():
    with pytest.raises(TypeError, match="boolean"):
        validate_attention_mask(torch.ones(1, 1, 2, 2))
    with pytest.raises(ValueError, match="2 to 4"):
        validate_attention_mask(torch.ones(1, 1, 1, 1, 1, dtype=torch.bool))
    with pytest.raises(ValueError, match="query dimension"):
        validate_attention_mask(
            torch.ones(1, 1, 3, 4, dtype=torch.bool),
            query_len=2,
            key_len=4,
        )
    with pytest.raises(ValueError, match="batch dimension"):
        validate_attention_mask(
            torch.ones(3, 1, 2, 4, dtype=torch.bool),
            batch_size=2,
            query_len=2,
            key_len=4,
        )


def test_self_attention_mask_uses_shared_semantics():
    tokens = torch.tensor([[1, 2, 0]])

    non_causal = self_attention_mask(tokens, causal=False)
    causal = self_attention_mask(tokens, causal=True)

    assert non_causal.shape == (1, 1, 1, 3)
    assert causal.shape == (1, 1, 3, 3)
    assert causal[0, 0].tolist() == [
        [True, False, False],
        [True, True, False],
        [True, True, False],
    ]
