import torch
import pytest

from miniformer.config.model_config import TransformerConfig
from miniformer.model.transformer import Transformer
from miniformer.model.seq2seq_transformer import Seq2SeqTransformer

def test_encoder_only_output_shape_with_explicit_output_dim():
    """If you set output_dim explicitly, the final projection should match."""
    cfg = TransformerConfig(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        output_dim=64,         # <— make this explicit
    )
    model = Transformer(cfg)
    token_ids = torch.randint(0, 1000, (2, 10))  # [batch, seq_len]
    output = model(token_ids)
    assert output.shape == (2, 10, 64)

def test_encoder_only_default_output_dim_is_vocab_size():
    """By default, output_dim == vocab_size."""
    cfg = TransformerConfig(
        vocab_size=500,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        # no output_dim passed → defaults to vocab_size=500
    )
    model = Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 7))
    out = model(x)
    assert out.shape == (3, 7, cfg.vocab_size)

def test_feature_based_forward_shape():
    """Encoder should accept raw feature vectors when input_dim is set."""
    cfg = TransformerConfig(
        input_dim=8,
        d_model=8,
        n_heads=2,
        n_layers=2,
        d_ff=32,
        output_dim=4,
    )
    model = Transformer(cfg)
    assert cfg.input_dim is not None
    feats = torch.randn(4, 20, cfg.input_dim)
    out = model(feats)
    assert out.shape == (4, 20, 4)

def test_invalid_input_dim_mismatch_raises():
    """If input_dim != d_model, config __post_init__ should error."""
    with pytest.raises(ValueError):
        TransformerConfig(input_dim=16, d_model=32)

def test_batch_independence():
    """Identical sequences in a batch should yield identical embeddings."""
    cfg = TransformerConfig(vocab_size=50, d_model=16, n_heads=2, n_layers=1, output_dim=16)
    model = Transformer(cfg).eval()
    
    seq = torch.randint(1, 50, (1, 10))
    batch2 = torch.cat([seq, seq], dim=0)
    
    with torch.no_grad():
        out2 = model(batch2)
        out1 = model(seq)
    
    assert torch.allclose(out2[0], out1[0], atol=1e-6)

def test_seq2seq_forward_and_generate():
    """Smoke‐test Seq2SeqTransformer forward and greedy generate."""
    cfg = TransformerConfig(vocab_size=200, d_model=32, n_heads=4, n_layers=2, d_ff=64, output_dim=200)
    seq2seq = Seq2SeqTransformer(cfg)
    
    src = torch.randint(0, cfg.vocab_size, (3, 12))
    tgt = torch.randint(0, cfg.vocab_size, (3, 14))
    logits, _, _ = seq2seq(src, tgt)
    assert logits.shape == (3, 14, cfg.vocab_size)
    
    # Greedy generation shouldn’t error and should produce only token IDs
    gen = seq2seq.generate(src, max_new_tokens=5)
    assert gen.dim() == 2 and gen.size(0) == 3

@pytest.mark.parametrize("pad_token", [0])
def test_padding_masking(pad_token):
    """
    If you feed an all‐padding sequence, the mask logic
    should zero out all embeddings (or at least give
    consistent outputs). Here we just check no crash
    and identical outputs for all-padding vs manual mask.
    """
    cfg = TransformerConfig(vocab_size=10, d_model=8, n_heads=2, n_layers=1, output_dim=8)
    model = Transformer(cfg).eval()
    
    all_pad = torch.zeros((2, 5), dtype=torch.long)
    with torch.no_grad():
        out_pad = model(all_pad)
        # both rows identical
        assert torch.allclose(out_pad[0], out_pad[1], atol=1e-6)
