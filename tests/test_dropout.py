import torch
import pytest
from miniformer.model.transformer import Transformer, TransformerConfig


def test_dropout_train_vs_eval_determinism():
    """Test that dropout behaves differently in train vs eval mode."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.5  # High dropout for noticeable effect
    )
    model = Transformer(config)
    
    x = torch.randint(0, 100, (2, 10))
    
    # In eval mode, outputs should be deterministic
    model.eval()
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)
    
    assert torch.allclose(output1, output2, atol=1e-6), "Eval mode should be deterministic"
    
    # In train mode, outputs should be different due to dropout
    model.train()
    torch.manual_seed(42)  # Reset seed
    with torch.no_grad():
        output3 = model(x)
    
    torch.manual_seed(43)  # Different seed
    with torch.no_grad():
        output4 = model(x)
    
    # Should be different (with high probability given 50% dropout)
    assert not torch.allclose(output3, output4, atol=1e-6), "Train mode should have stochastic dropout"


def test_dropout_consistency_across_layers():
    """Test that dropout is applied consistently across all layers."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=8,
        n_layers=3,
        dropout=0.3
    )
    model = Transformer(config)
    
    x = torch.randint(0, 100, (4, 15))
    
    # In eval mode, multiple passes should be identical
    model.eval()
    outputs = []
    for _ in range(3):
        with torch.no_grad():
            outputs.append(model(x))
    
    for i in range(1, len(outputs)):
        assert torch.allclose(outputs[0], outputs[i], atol=1e-6), f"Eval pass {i} differs from pass 0"


def test_dropout_preserves_shape():
    """Test that dropout doesn't change tensor shapes."""
    config = TransformerConfig(
        vocab_size=50,
        d_model=48,
        n_heads=6,
        n_layers=2,
        dropout=0.8  # Very high dropout
    )
    model = Transformer(config)
    
    x = torch.randint(0, 50, (3, 12))
    
    model.train()
    output = model(x)
    
    assert output.shape == (3, 12, 48), "Dropout should not change output shape"
    assert torch.isfinite(output).all(), "Dropout should not produce non-finite values"


def test_zero_dropout_equivalence():
    """Test that zero dropout makes train and eval modes equivalent."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2,
        dropout=0.0  # No dropout
    )
    model = Transformer(config)
    
    x = torch.randint(0, 100, (2, 8))
    
    # Get output in eval mode
    model.eval()
    with torch.no_grad():
        eval_output = model(x)
    
    # Get output in train mode
    model.train()
    with torch.no_grad():
        train_output = model(x)
    
    # Should be identical when dropout=0
    assert torch.allclose(eval_output, train_output, atol=1e-6), "Zero dropout should make train/eval equivalent"