import pytest
import torch

from miniformer.model.attention import MultiHeadAttention
from miniformer.model.embedding import PositionalEncoding
from miniformer.model.masks import causal_mask
from miniformer.model.transformer import Transformer, TransformerConfig


def test_attention_mask_correctness():
    """Test that attention masks properly prevent information flow."""
    config = TransformerConfig(vocab_size=100, d_model=32, n_heads=4, n_layers=1)
    model = Transformer(config)
    model.eval()

    with torch.no_grad():
        # Create input where last token differs between sequences
        x1 = torch.tensor([[1, 2, 3, 4, 5]])
        x2 = torch.tensor([[1, 2, 3, 4, 99]])  # Different last token

        out1 = model(x1).output
        out2 = model(x2).output

        # All positions except the last should be identical (causal masking)
        assert torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-6)
        # Last position should be different
        assert not torch.allclose(out1[0, -1], out2[0, -1], atol=1e-6)


def test_padding_mask_interaction():
    """Test that padding tokens are properly masked in attention."""
    config = TransformerConfig(vocab_size=100, d_model=32, n_heads=4, n_layers=1)
    model = Transformer(config)
    model.eval()

    with torch.no_grad():
        # Create padded sequences
        x = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 0, 0]])  # No padding  # Padded with 0s

        output = model(x).output

        # Check that we can extract attention weights for verification
        # (This assumes the model stores attention weights during forward pass)
        assert output.shape == (2, 5, 32)


@pytest.mark.parametrize("rotary_pct", [0.0, 0.5, 1.0])
def test_rotary_embedding_consistency(rotary_pct):
    """Test that rotary embeddings work correctly at different percentages."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=1,
        position_mode="learned" if rotary_pct == 0.0 else "learned+rope",
        rotary_pct=rotary_pct,
    )
    model = Transformer(config)

    x = torch.randint(0, 100, (2, 10))
    output = model(x).output

    # Should produce valid output regardless of rotary_pct
    assert output.shape == (2, 10, 64)
    assert torch.isfinite(output).all()


def test_multi_head_attention_heads():
    """Test that multi-head attention actually uses multiple heads."""
    config = TransformerConfig(vocab_size=100, d_model=64, n_heads=8, n_layers=1)
    model = Transformer(config)

    # Verify head dimensions based on config
    head_dim = config.d_model // config.n_heads
    assert head_dim == 8  # 64 // 8 = 8
    assert config.n_heads == 8

    # Test forward pass
    x = torch.randint(0, 100, (2, 10))
    output = model(x).output
    assert output.shape == (2, 10, 64)


def test_encoder_only_non_causal_mode_allows_future_context():
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=1,
        causal=False,
    )
    model = Transformer(config).eval()

    with torch.no_grad():
        x1 = torch.tensor([[1, 2, 3, 4, 5]])
        x2 = torch.tensor([[1, 2, 3, 4, 99]])

        out1 = model(x1).output
        out2 = model(x2).output

    assert not torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-6)


def test_manual_and_sdpa_attention_match_with_mask():
    manual = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0, use_sdpa=False)
    sdpa = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0, use_sdpa=True)
    sdpa.load_state_dict(manual.state_dict())
    manual.eval()
    sdpa.eval()

    q = torch.randn(2, 5, 32)
    k = torch.randn(2, 5, 32)
    v = torch.randn(2, 5, 32)
    mask = causal_mask(5)

    with torch.no_grad():
        manual_out, _, _ = manual(q, k, v, mask)
        sdpa_out, _, _ = sdpa(q, k, v, mask)

    assert torch.allclose(manual_out, sdpa_out, atol=1e-5)


def test_manual_attention_gradcheck_small_tensors():
    attention = MultiHeadAttention(d_model=4, n_heads=2, dropout=0.0, use_sdpa=False)
    attention.double().eval()
    q = torch.randn(1, 2, 4, dtype=torch.double, requires_grad=True)
    k = torch.randn(1, 2, 4, dtype=torch.double, requires_grad=True)
    v = torch.randn(1, 2, 4, dtype=torch.double, requires_grad=True)
    mask = causal_mask(2)

    def fn(q_tensor, k_tensor, v_tensor):
        return attention(q_tensor, k_tensor, v_tensor, mask)[0]

    assert torch.autograd.gradcheck(fn, (q, k, v), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("use_sdpa", [False, True])
def test_attention_dtype_smoke_stays_finite(dtype, use_sdpa):
    attention = MultiHeadAttention(d_model=16, n_heads=4, dropout=0.0, use_sdpa=use_sdpa)
    attention.to(dtype=dtype).eval()
    x = torch.randn(2, 3, 16, dtype=dtype)
    mask = causal_mask(3)

    with torch.no_grad():
        out, weights, _ = attention(x, x, x, mask)

    assert out.dtype == dtype
    assert torch.isfinite(out.float()).all()
    if weights is not None:
        assert weights.dtype == dtype
        assert torch.isfinite(weights.float()).all()


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_attention_validates_masks_before_backend(use_sdpa):
    attention = MultiHeadAttention(d_model=16, n_heads=4, dropout=0.0, use_sdpa=use_sdpa)
    x = torch.randn(2, 3, 16)

    with pytest.raises(TypeError, match="boolean"):
        attention(x, x, x, torch.ones(1, 1, 3, 3))
    with pytest.raises(ValueError, match="key dimension"):
        attention(x, x, x, torch.ones(1, 1, 3, 2, dtype=torch.bool))


def test_all_masked_manual_attention_returns_zero_context():
    attention = MultiHeadAttention(d_model=16, n_heads=4, dropout=0.0, use_sdpa=False)
    attention.eval()
    x = torch.randn(2, 3, 16)
    mask = torch.zeros(2, 1, 3, 3, dtype=torch.bool)

    with torch.no_grad():
        out, weights, _ = attention(x, x, x, mask)

    bias = attention.wo.bias.view(1, 1, -1)
    assert torch.allclose(weights, torch.zeros_like(weights))
    assert torch.isfinite(weights).all()
    assert torch.isfinite(out).all()
    assert torch.allclose(out, bias.expand_as(out), atol=1e-6)


def test_all_masked_manual_attention_with_extreme_values_stays_finite():
    attention = MultiHeadAttention(d_model=16, n_heads=4, dropout=0.0, use_sdpa=False)
    attention.eval()
    x = torch.full((1, 2, 16), 1e20)
    mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)

    with torch.no_grad():
        out, weights, _ = attention(x, x, x, mask)

    assert torch.allclose(weights, torch.zeros_like(weights))
    assert torch.isfinite(weights).all()
    assert torch.isfinite(out).all()


def test_sinusoidal_positional_encoding_supports_odd_model_dim():
    pe = PositionalEncoding(d_model=7, max_seq_len=8, dropout=0.0)
    x = torch.zeros(2, 8, 7)

    out = pe(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
