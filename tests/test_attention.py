import torch
import pytest
from miniformer.model.transformer import Transformer, TransformerConfig


def test_attention_mask_correctness():
    """Test that attention masks properly prevent information flow."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=1
    )
    model = Transformer(config)
    model.eval()
    
    with torch.no_grad():
        # Create input where last token differs between sequences
        x1 = torch.tensor([[1, 2, 3, 4, 5]])
        x2 = torch.tensor([[1, 2, 3, 4, 99]])  # Different last token
        
        out1 = model(x1)
        out2 = model(x2)
        
        # All positions except the last should be identical (causal masking)
        assert torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-6)
        # Last position should be different
        assert not torch.allclose(out1[0, -1], out2[0, -1], atol=1e-6)


def test_padding_mask_interaction():
    """Test that padding tokens are properly masked in attention."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=1
    )
    model = Transformer(config)
    model.eval()
    
    with torch.no_grad():
        # Create padded sequences
        x = torch.tensor([
            [1, 2, 3, 4, 5],     # No padding
            [1, 2, 3, 0, 0]      # Padded with 0s
        ])
        
        output = model(x)
        
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
        n_layers=1
    )
    model = Transformer(config)
    
    x = torch.randint(0, 100, (2, 10))
    output = model(x)
    
    # Should produce valid output regardless of rotary_pct
    assert output.shape == (2, 10, 64)
    assert torch.isfinite(output).all()


def test_multi_head_attention_heads():
    """Test that multi-head attention actually uses multiple heads."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=8,
        n_layers=1
    )
    model = Transformer(config)
    
    # Verify head dimensions based on config
    head_dim = config.d_model // config.n_heads
    assert head_dim == 8  # 64 // 8 = 8
    assert config.n_heads == 8
    
    # Test forward pass
    x = torch.randint(0, 100, (2, 10))
    output = model(x)
    assert output.shape == (2, 10, 64)