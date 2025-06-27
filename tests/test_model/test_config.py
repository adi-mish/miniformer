import pytest
import torch
from miniformer.model.transformer import Transformer, TransformerConfig


@pytest.mark.parametrize("d_model,n_heads", [
    (32, 4),
    (48, 3), 
    (64, 8),
    (128, 16)
])
def test_valid_config_combinations(d_model, n_heads):
    """Test that valid d_model/n_heads combinations work correctly."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=2,
        d_ff=d_model * 4
    )
    model = Transformer(config)
    
    # Test forward pass works
    x = torch.randint(0, 100, (2, 10))
    output = model(x)
    assert output.shape == (2, 10, d_model)


def test_invalid_n_heads_not_divisible():
    """Test that n_heads must divide d_model evenly."""
    with pytest.raises(ValueError, match="d_model.*divisible.*n_heads"):
        TransformerConfig(
            vocab_size=100,
            d_model=64,
            n_heads=5,  # 64 not divisible by 5
            n_layers=2
        )


def test_invalid_activation():
    """Test that invalid activation functions are rejected."""
    with pytest.raises(ValueError, match="Unknown activation"):
        TransformerConfig(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            activation="invalid_activation"
        )


def test_input_dim_projection():
    """Test that input_dim != d_model creates proper projection layer."""
    config = TransformerConfig(
        input_dim=5,
        d_model=32,
        n_heads=4,
        n_layers=2,
        output_dim=1
    )
    model = Transformer(config)
    
    # Should have input projection
    assert hasattr(model, 'input_projection')
    assert isinstance(model.input_projection, torch.nn.Linear)
    assert model.input_projection.in_features == 5
    assert model.input_projection.out_features == 32
    
    # Test forward pass
    x = torch.randn(2, 10, 5)
    output = model(x)
    assert output.shape == (2, 10, 1)