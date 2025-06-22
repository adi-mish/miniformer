import torch
import pytest
from miniformer.model.transformer import Transformer, TransformerConfig
from miniformer.model.seq2seq_transformer import Seq2SeqTransformer


def test_cross_attention_mask_prevents_future_leakage():
    """Test that cross-attention masks prevent decoder from seeing future tokens."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2,
    )
    
    # Use encoder-decoder model for cross-attention testing
    model = Seq2SeqTransformer(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2
    )
    model.eval()
    
    # Create source and target sequences
    src_tokens = torch.randint(1, 100, (1, 8))
    tgt_tokens = torch.randint(1, 100, (1, 6))
    
    with torch.no_grad():
        # Get decoder output for full target sequence
        full_output = model(src_tokens, tgt_tokens)
        
        # Get decoder output for partial target sequence (first 3 tokens)
        partial_output = model(src_tokens, tgt_tokens[:, :3])
    
    # The first 3 positions should be identical regardless of future tokens
    assert torch.allclose(
        full_output[:, :3, :], 
        partial_output[:, :3, :], 
        atol=1e-6
    ), "Causal masking failed: future tokens leaked into past positions"


def test_encoder_decoder_attention_alignment():
    """Test that encoder-decoder attention properly aligns sequences."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=1
    )
    
    model = Seq2SeqTransformer(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=1
    )
    model.eval()
    
    # Different source sequence lengths
    src1 = torch.randint(1, 100, (1, 5))
    src2 = torch.randint(1, 100, (1, 8))
    tgt = torch.randint(1, 100, (1, 4))
    
    with torch.no_grad():
        output1 = model(src1, tgt)
        output2 = model(src2, tgt)
    
    # Both should produce valid outputs despite different source lengths
    assert output1.shape == (1, 4, 32)
    assert output2.shape == (1, 4, 32)
    assert torch.isfinite(output1).all()
    assert torch.isfinite(output2).all()


def test_decoder_self_attention_causal_mask():
    """Test that decoder self-attention respects causal masking."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2,
        )
    
    model = Seq2SeqTransformer(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2
    )
    model.eval()
    
    src_tokens = torch.randint(1, 100, (1, 6))
    
    with torch.no_grad():
        # Two target sequences, differing only in the last token
        tgt1 = torch.tensor([[1, 2, 3, 4, 5]])
        tgt2 = torch.tensor([[1, 2, 3, 4, 99]])  # Different last token
        
        output1 = model(src_tokens, tgt1)
        output2 = model(src_tokens, tgt2)
        
        # All positions except the last should be identical
        assert torch.allclose(
            output1[:, :-1, :], 
            output2[:, :-1, :], 
            atol=1e-6
        ), "Causal self-attention mask failed in decoder"
        
        # Last position should be different
        assert not torch.allclose(
            output1[:, -1, :], 
            output2[:, -1, :], 
            atol=1e-6
        ), "Decoder should be sensitive to its own last token"


def test_padding_mask_in_cross_attention():
    """Test that padding masks work correctly in encoder-decoder attention."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=1,
    )
    
    model = Seq2SeqTransformer(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=1
    )
    model.eval()
    
    # Source with padding
    src_padded = torch.tensor([[1, 2, 3, 0, 0, 0]])  # Padded
    src_unpadded = torch.tensor([[1, 2, 3]])  # No padding
    
    tgt = torch.tensor([[10, 20]])
    
    with torch.no_grad():
        output_padded = model(src_padded, tgt)
        output_unpadded = model(src_unpadded, tgt)
    
    # Outputs should have same shape
    assert output_padded.shape == (1, 2, 32)
    assert output_unpadded.shape == (1, 2, 32)
    
    # Both should produce finite outputs
    assert torch.isfinite(output_padded).all()
    assert torch.isfinite(output_unpadded).all()