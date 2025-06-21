import torch
import pytest
from miniformer.model.transformer import Transformer, TransformerConfig

def test_model_output_shape_nlp():
    """Tests the output shape for an NLP-style model."""
    model = Transformer(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=256,
    )
    token_ids = torch.randint(0, 1000, (2, 10))  # [batch, seq_len]
    output = model(token_ids)
    assert output.shape == (2, 10, 64) # [batch, seq_len, d_model]

def test_model_output_shape_features():
    """Tests the output shape for a feature-based model."""
    config = TransformerConfig(
        input_dim=5,
        d_model=32,
        output_dim=1,
        n_heads=2,
        n_layers=2,
        d_ff=64
    )
    model = Transformer(config)
    features = torch.randn(2, 24, 5)  # [batch, seq_len, features]
    output = model(features)
    assert output.shape == (2, 24, 1) # [batch, seq_len, output_dim]

def test_batch_independence():
    """Tests that batch items are processed independently."""
    model = Transformer(vocab_size=100, d_model=16, n_heads=2, n_layers=1)
    
    # Input with two identical sequences
    seq = torch.randint(0, 100, (1, 10))
    batch_double = torch.cat([seq, seq], dim=0)
    
    # Input with one of the sequences
    batch_single = seq
    
    model.eval() # Use eval mode to disable dropout for consistent output
    with torch.no_grad():
        output_double = model(batch_double)
        output_single = model(batch_single)

    # The output for the first item in the batch should be identical to the output of a single-item batch
    assert torch.allclose(output_double[0], output_single[0], atol=1e-6)

def test_causal_masking():
    """
    Tests that the model respects the causal mask for autoregressive tasks.
    This test assumes that providing a `vocab_size` to the Transformer
    implicitly enables causal masking for language modeling.
    """
    # is_causal=True might be needed if the model doesn't infer it.
    # e.g. model = Transformer(..., is_causal=True)
    model = Transformer(vocab_size=100, d_model=16, n_heads=2, n_layers=1)
    
    model.eval()
    with torch.no_grad():
        # Two sequences, the second is identical to the first except for the last token
        seq1 = torch.randint(1, 100, (1, 10))
        seq2 = seq1.clone()
        seq2[0, -1] = 0 # Change the last token
        
        batch = torch.cat([seq1, seq2], dim=0)
        output = model(batch)

    # The outputs for the two sequences should be identical for all positions *before* the last one
    assert torch.allclose(output[0, :-1, :], output[1, :-1, :], atol=1e-6)
    
    # The outputs at the last position should be different
    assert not torch.allclose(output[0, -1, :], output[1, -1, :])
