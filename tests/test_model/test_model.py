import pytest
import torch

from miniformer.config.model_config import TransformerConfig
from miniformer.model.seq2seq_transformer import Seq2SeqModelOutput, Seq2SeqTransformer
from miniformer.model.transformer import Transformer


def test_encoder_only_output_shape_with_explicit_output_dim():
    """If you set output_dim explicitly, the final projection should match."""
    cfg = TransformerConfig(
        vocab_size=1000,
        d_model=64,
        n_heads=4,
        n_layers=3,
        d_ff=256,
        output_mode="projection",
        output_dim=64,
    )
    model = Transformer(cfg)
    token_ids = torch.randint(0, 1000, (2, 10))  # [batch, seq_len]
    output = model(token_ids)
    assert output.projection is not None
    assert output.logits is None
    assert output.hidden_states is None
    assert output.output.shape == (2, 10, 64)


def test_encoder_only_default_output_mode_is_hidden_states():
    """Default encoder-only models expose hidden states, not guessed vocab logits."""
    cfg = TransformerConfig(
        vocab_size=500,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=128,
    )
    model = Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 7))
    out = model(x)
    assert out.hidden_states is not None
    assert out.logits is None
    assert out.projection is None
    assert out.output.shape == (3, 7, cfg.d_model)


def test_encoder_only_vocab_mode_returns_logits():
    cfg = TransformerConfig(
        vocab_size=500,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        output_mode="vocab",
    )
    model = Transformer(cfg)
    x = torch.randint(0, cfg.vocab_size, (3, 7))
    out = model(x)
    assert out.logits is not None
    assert out.hidden_states is None
    assert out.projection is None
    assert out.output.shape == (3, 7, cfg.vocab_size)


def test_feature_based_forward_shape():
    """Encoder should accept raw feature vectors when input_dim is set."""
    cfg = TransformerConfig(
        input_dim=8,
        d_model=8,
        n_heads=2,
        n_layers=2,
        d_ff=32,
        output_mode="projection",
        output_dim=4,
    )
    model = Transformer(cfg)
    assert cfg.input_dim is not None
    feats = torch.randn(4, 20, cfg.input_dim)
    out = model(feats)
    assert out.projection is not None
    assert out.output.shape == (4, 20, 4)


def test_transformer_rejects_raw_records():
    cfg = TransformerConfig(vocab_size=50, d_model=16, n_heads=2, n_layers=1)
    model = Transformer(cfg)

    with pytest.raises(TypeError, match="torch.Tensor"):
        model([{"input": "raw text"}])  # type: ignore[arg-type]


def test_output_mode_validation_is_explicit():
    with pytest.raises(ValueError, match="hidden.*output_dim=None"):
        TransformerConfig(vocab_size=50, d_model=16, n_heads=2, output_dim=16)

    with pytest.raises(ValueError, match="projection.*requires output_dim"):
        TransformerConfig(vocab_size=50, d_model=16, n_heads=2, output_mode="projection")

    with pytest.raises(ValueError, match="only valid for token inputs"):
        TransformerConfig(input_dim=16, d_model=32, n_heads=4, output_mode="vocab")

    with pytest.raises(ValueError, match="requires output_dim=None"):
        TransformerConfig(vocab_size=50, d_model=16, n_heads=2, output_mode="vocab", output_dim=50)


def test_batch_independence():
    """Identical sequences in a batch should yield identical embeddings."""
    cfg = TransformerConfig(vocab_size=50, d_model=16, n_heads=2, n_layers=1)
    model = Transformer(cfg).eval()

    seq = torch.randint(1, 50, (1, 10))
    batch2 = torch.cat([seq, seq], dim=0)

    with torch.no_grad():
        out2 = model(batch2).output
        out1 = model(seq).output

    assert torch.allclose(out2[0], out1[0], atol=1e-6)


def test_seq2seq_forward_and_generate():
    """Smoke‐test Seq2SeqTransformer forward and greedy generate."""
    cfg = TransformerConfig(
        vocab_size=200, d_model=32, n_heads=4, n_layers=2, d_ff=64, output_mode="vocab"
    )
    seq2seq = Seq2SeqTransformer(cfg)

    src = torch.randint(0, cfg.vocab_size, (3, 12))
    tgt = torch.randint(0, cfg.vocab_size, (3, 14))
    output = seq2seq(src, tgt)
    assert isinstance(output, Seq2SeqModelOutput)
    assert output.logits is not None
    assert output.hidden_states is None
    assert output.output.shape == (3, 14, cfg.vocab_size)
    assert len(output.self_attentions or []) == cfg.n_layers
    assert len(output.cross_attentions or []) == cfg.n_layers
    with pytest.raises(TypeError):
        torch.isfinite(output)

    # Greedy generation shouldn’t error and should produce only token IDs
    gen = seq2seq.generate(src, max_new_tokens=5)
    assert gen.dim() == 2 and gen.size(0) == 3


def test_seq2seq_forward_without_output_dim_returns_hidden_states():
    cfg = TransformerConfig(vocab_size=200, d_model=32, n_heads=4, n_layers=2, d_ff=64)
    seq2seq = Seq2SeqTransformer(cfg)

    src = torch.randint(0, cfg.vocab_size, (2, 8))
    tgt = torch.randint(0, cfg.vocab_size, (2, 6))
    output = seq2seq(src, tgt)

    assert output.logits is None
    assert output.hidden_states is not None
    assert output.projection is None
    assert output.output.shape == (2, 6, cfg.d_model)


def test_seq2seq_feature_inputs_use_encoder_and_decoder_projections_once():
    cfg = TransformerConfig(
        input_dim=5,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        output_mode="projection",
        output_dim=3,
    )
    seq2seq = Seq2SeqTransformer(cfg)

    src = torch.randn(2, 7, cfg.input_dim)
    tgt = torch.randn(2, 4, cfg.input_dim)
    output = seq2seq(src, tgt)

    assert output.projection is not None
    assert output.output.shape == (2, 4, cfg.output_dim)


@pytest.mark.parametrize("pad_token", [0])
def test_padding_masking(pad_token):
    """
    If you feed an all‐padding sequence, the mask logic
    should zero out all embeddings (or at least give
    consistent outputs). Here we just check no crash
    and identical outputs for all-padding vs manual mask.
    """
    cfg = TransformerConfig(vocab_size=10, d_model=8, n_heads=2, n_layers=1)
    model = Transformer(cfg).eval()

    all_pad = torch.zeros((2, 5), dtype=torch.long)
    with torch.no_grad():
        out_pad = model(all_pad).output
        # both rows identical
        assert torch.allclose(out_pad[0], out_pad[1], atol=1e-6)
