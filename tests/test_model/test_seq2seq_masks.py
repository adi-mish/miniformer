import pytest
import torch

from miniformer.model.masks import padding_mask
from miniformer.model.seq2seq_transformer import Seq2SeqTransformer
from miniformer.model.transformer import TransformerConfig


def test_seq2seq_forward_causal_mask_prevents_future_target_leakage():
    """The decoder must not let later target tokens affect earlier target positions."""
    model = Seq2SeqTransformer(vocab_size=100, d_model=32, n_heads=4, n_layers=2)
    model.eval()

    # Create source and target sequences
    src_tokens = torch.randint(1, 100, (1, 8))
    tgt_tokens = torch.randint(1, 100, (1, 6))

    with torch.no_grad():
        # Get decoder output for full target sequence
        full_output = model(src_tokens, tgt_tokens).output

        # Get decoder output for partial target sequence (first 3 tokens)
        partial_output = model(src_tokens, tgt_tokens[:, :3]).output

    # The first 3 positions should be identical regardless of future tokens
    assert torch.allclose(
        full_output[:, :3, :], partial_output[:, :3, :], atol=1e-6
    ), "Causal masking failed: future tokens leaked into past positions"


def test_encoder_decoder_attention_alignment():
    """Test that encoder-decoder attention properly aligns sequences."""
    model = Seq2SeqTransformer(vocab_size=100, d_model=32, n_heads=4, n_layers=1)
    model.eval()

    # Different source sequence lengths
    src1 = torch.randint(1, 100, (1, 5))
    src2 = torch.randint(1, 100, (1, 8))
    tgt = torch.randint(1, 100, (1, 4))

    with torch.no_grad():
        output1 = model(src1, tgt).output
        output2 = model(src2, tgt).output

    # Both should produce valid outputs despite different source lengths
    assert output1.shape == (1, 4, 32)
    assert output2.shape == (1, 4, 32)
    assert torch.isfinite(output1).all()
    assert torch.isfinite(output2).all()


def test_decoder_self_attention_causal_mask():
    """Test that decoder self-attention respects causal masking."""
    model = Seq2SeqTransformer(vocab_size=100, d_model=32, n_heads=4, n_layers=2)
    model.eval()

    src_tokens = torch.randint(1, 100, (1, 6))

    with torch.no_grad():
        # Two target sequences, differing only in the last token
        tgt1 = torch.tensor([[1, 2, 3, 4, 5]])
        tgt2 = torch.tensor([[1, 2, 3, 4, 99]])  # Different last token

        output1 = model(src_tokens, tgt1).output
        output2 = model(src_tokens, tgt2).output

        # All positions except the last should be identical
        assert torch.allclose(
            output1[:, :-1, :], output2[:, :-1, :], atol=1e-6
        ), "Causal self-attention mask failed in decoder"

        # Last position should be different
        assert not torch.allclose(
            output1[:, -1, :], output2[:, -1, :], atol=1e-6
        ), "Decoder should be sensitive to its own last token"


def test_padding_mask_in_cross_attention():
    """Test that padding masks work correctly in encoder-decoder attention."""
    model = Seq2SeqTransformer(vocab_size=100, d_model=32, n_heads=4, n_layers=1)
    model.eval()

    # Source with padding
    src_padded = torch.tensor([[1, 2, 3, 0, 0, 0]])  # Padded
    src_unpadded = torch.tensor([[1, 2, 3]])  # No padding

    tgt = torch.tensor([[10, 20]])

    with torch.no_grad():
        output_padded = model(src_padded, tgt).output
        output_unpadded = model(src_unpadded, tgt).output

    # Outputs should have same shape
    assert output_padded.shape == (1, 2, 32)
    assert output_unpadded.shape == (1, 2, 32)

    # Both should produce finite outputs
    assert torch.isfinite(output_padded).all()
    assert torch.isfinite(output_unpadded).all()


def test_decoder_cache_matches_full_pass_for_chunked_targets():
    config = TransformerConfig(
        vocab_size=64,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        dropout=0.0,
        output_mode="vocab",
        position_mode="learned+rope",
        rotary_pct=0.5,
    )
    model = Seq2SeqTransformer(config).eval()

    src = torch.randint(1, config.vocab_size, (2, 5))
    tgt = torch.randint(1, config.vocab_size, (2, 6))
    src_mask = padding_mask(src)

    with torch.no_grad():
        memory = model.encoder(src, src_mask)
        full_decoder_output = model.decoder(
            tgt,
            memory,
            cross_attn_mask=src_mask,
            use_causal_mask=True,
        )
        full_out = full_decoder_output.output

        past_key_values = None
        cached_chunks = []
        for start, end in [(0, 2), (2, 5), (5, 6)]:
            chunk_decoder_output = model.decoder(
                tgt[:, start:end],
                memory,
                cross_attn_mask=src_mask,
                use_causal_mask=True,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = chunk_decoder_output.past_key_values
            cached_chunks.append(chunk_decoder_output.output)

    cached_out = torch.cat(cached_chunks, dim=1)
    assert torch.allclose(full_out, cached_out, atol=1e-5)


def test_seq2seq_generate_rejects_non_vocab_output_head():
    config = TransformerConfig(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        output_mode="projection",
        output_dim=4,
    )
    model = Seq2SeqTransformer(config).eval()
    src = torch.randint(1, config.vocab_size, (1, 3))

    with pytest.raises(RuntimeError, match="requires output_mode='vocab'"):
        model.generate(src, max_new_tokens=1)
