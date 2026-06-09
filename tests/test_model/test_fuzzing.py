import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from miniformer.model.transformer import Transformer, TransformerConfig


@given(batch=st.integers(1, 4), seq=st.integers(1, 32), vocab=st.integers(10, 100))
@settings(max_examples=20, deadline=5000)  # Limit examples for faster testing
def test_encoder_shapes_fuzz(batch, seq, vocab):
    """Property-based test for encoder output shapes with random inputs."""
    config = TransformerConfig(vocab_size=vocab, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    model = Transformer(config)

    x = torch.randint(0, vocab, (batch, seq))
    output = model(x).output

    assert output.shape == (batch, seq, config.d_model)
    assert torch.isfinite(output).all()


@given(batch=st.integers(1, 4), seq=st.integers(1, 32), features=st.integers(2, 16))
@settings(max_examples=20, deadline=5000)
def test_feature_input_shapes_fuzz(batch, seq, features):
    """Property-based test for feature input handling."""
    config = TransformerConfig(
        input_dim=features,
        d_model=32,
        n_heads=4,
        n_layers=2,
        output_mode="projection",
        output_dim=1,
    )
    model = Transformer(config)

    x = torch.randn(batch, seq, features)
    output = model(x).output

    assert output.shape == (batch, seq, 1)
    assert torch.isfinite(output).all()


@given(
    d_model=st.integers(16, 128).filter(
        lambda x: x % 4 == 0
    ),  # Ensure divisible by common head counts
    n_heads=st.sampled_from([2, 4, 8, 16]),
)
@settings(max_examples=15, deadline=5000)
def test_attention_dimension_validation_fuzz(d_model, n_heads):
    """Property-based test for valid attention dimensions."""
    # Only test valid combinations
    if d_model % n_heads != 0:
        pytest.skip("Invalid d_model/n_heads combination")

    config = TransformerConfig(vocab_size=50, d_model=d_model, n_heads=n_heads, n_layers=1)
    model = Transformer(config)

    x = torch.randint(0, 50, (2, 8))
    output = model(x).output

    assert output.shape == (2, 8, d_model)
    assert torch.isfinite(output).all()


@given(max_seq_len=st.integers(1, 32))
@settings(max_examples=15, deadline=5000)
def test_sequence_length_within_config_limit_fuzz(max_seq_len):
    """Property-based test for sequence length handling."""
    config = TransformerConfig(
        vocab_size=100, d_model=32, n_heads=4, n_layers=1, max_seq_len=max_seq_len
    )
    model = Transformer(config)
    x = torch.randint(0, 100, (1, max_seq_len))

    output = model(x).output

    assert output.shape == (1, max_seq_len, 32)
    assert torch.isfinite(output).all()


def test_sequence_length_exceeding_config_limit_raises():
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=1,
        max_seq_len=4,
    )
    model = Transformer(config)
    x = torch.randint(0, 100, (1, 5))

    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(x)


def test_edge_case_single_token():
    """Test handling of single token sequences."""
    config = TransformerConfig(vocab_size=100, d_model=32, n_heads=4, n_layers=2)
    model = Transformer(config)

    # Single token input
    x = torch.randint(0, 100, (1, 1))
    output = model(x).output

    assert output.shape == (1, 1, 32)
    assert torch.isfinite(output).all()


def test_edge_case_large_batch():
    """Test handling of large batch sizes."""
    config = TransformerConfig(vocab_size=100, d_model=32, n_heads=4, n_layers=2)
    model = Transformer(config)

    # Large batch
    x = torch.randint(0, 100, (32, 10))
    output = model(x).output

    assert output.shape == (32, 10, 32)
    assert torch.isfinite(output).all()


def test_numerical_stability_extreme_values():
    """Test model stability with extreme input values."""
    config = TransformerConfig(
        input_dim=4,
        d_model=32,
        n_heads=4,
        n_layers=2,
        output_mode="projection",
        output_dim=1,
    )
    model = Transformer(config)
    model.eval()

    # Test with various extreme values
    test_cases = [
        torch.zeros(2, 10, 4),  # All zeros
        torch.ones(2, 10, 4) * 100,  # Large positive values
        torch.ones(2, 10, 4) * -100,  # Large negative values
        torch.randn(2, 10, 4) * 10,  # High variance
    ]

    for x in test_cases:
        with torch.no_grad():
            output = model(x).output
            assert torch.isfinite(
                output
            ).all(), f"Non-finite output for input with mean {x.mean():.2f}"
            assert not torch.isnan(output).any(), f"NaN output for input with mean {x.mean():.2f}"
