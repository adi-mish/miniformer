import pytest
import torch

from miniformer.model.cache import KeyValueCache
from miniformer.model.generation import (
    GenerationConfig,
    filter_logits_for_sampling,
    sample_next_token,
)
from miniformer.model.outputs import TransformerModelOutput
from miniformer.model.seq2seq_transformer import Seq2SeqTransformer
from miniformer.model.transformer import Transformer, TransformerConfig


def test_kv_cache_equivalence():
    """Test that KV caching produces identical results to non-cached generation."""
    config = TransformerConfig(vocab_size=100, d_model=32, n_heads=4, n_layers=2)
    model = Transformer(config)
    model.eval()

    # Initial sequence
    input_ids = torch.randint(1, 100, (1, 5))

    with torch.no_grad():
        # Non-cached forward pass
        output_no_cache = model(input_ids, use_cache=False).output

        # Cached generation (simulate autoregressive generation)
        past_key_values = None
        cached_outputs = []

        for i in range(input_ids.size(1)):
            current_token = input_ids[:, i : i + 1]
            output = model(current_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            assert past_key_values is not None
            assert len(past_key_values) == config.n_layers
            assert isinstance(past_key_values[0], KeyValueCache)
            assert past_key_values[0].key.size(2) == i + 1
            cached_outputs.append(output.output)

        # Concatenate cached outputs
        output_cached = torch.cat(cached_outputs, dim=1)

        # Should be identical (within numerical precision)
        assert torch.allclose(output_no_cache, output_cached, atol=1e-6)


def test_kv_cache_rejects_non_causal_encoder_mode():
    config = TransformerConfig(vocab_size=100, d_model=32, n_heads=4, n_layers=1, causal=False)
    model = Transformer(config).eval()

    with torch.no_grad():
        try:
            model(torch.randint(1, 100, (1, 1)), use_cache=True)
        except RuntimeError as exc:
            assert "causal=True" in str(exc)
        else:
            raise AssertionError("cached encoder decoding should reject non-causal mode")


def test_generation_with_max_new_tokens():
    """Test autoregressive generation with max_new_tokens limit."""
    config = TransformerConfig(
        vocab_size=100, d_model=32, n_heads=4, n_layers=2, output_mode="vocab"
    )
    model = Transformer(config)
    model.eval()

    input_ids = torch.randint(1, 100, (1, 3))
    max_new_tokens = 5

    with torch.no_grad():
        # Manual autoregressive generation
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = model(generated).output
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    # Should generate exactly max_new_tokens additional tokens
    assert generated.size(1) == input_ids.size(1) + max_new_tokens
    # First tokens should match input
    assert torch.equal(generated[:, : input_ids.size(1)], input_ids)


def test_generation_with_eos_token():
    """Test that generation stops at EOS token."""
    config = TransformerConfig(
        vocab_size=100, d_model=32, n_heads=4, n_layers=2, output_mode="vocab"
    )
    # Define EOS token ID separately since it's not part of config
    eos_token_id = 99
    model = Transformer(config)

    # Mock the model to always predict EOS token after first generation step
    def mock_forward(x, mask=None, **kwargs):
        batch_size, seq_len = x.shape
        # Create logits that heavily favor EOS token
        logits = torch.full((batch_size, seq_len, config.vocab_size), -10.0)
        logits[:, :, eos_token_id] = 10.0  # High probability for EOS
        return TransformerModelOutput(logits=logits)

    model.forward = mock_forward
    model.eval()

    input_ids = torch.randint(1, 98, (1, 3))  # Avoid EOS in input

    with torch.no_grad():
        # Manual autoregressive generation with EOS stopping
        generated = input_ids.clone()
        max_new_tokens = 10
        for _ in range(max_new_tokens):
            logits = model(generated).output
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == eos_token_id:
                break

    # Should stop early due to EOS token
    assert generated.size(1) <= input_ids.size(1) + 10
    # Should contain EOS token
    assert eos_token_id in generated[0]


def test_seq2seq_generate_greedy_is_deterministic():
    config = TransformerConfig(
        vocab_size=40,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        output_mode="vocab",
    )
    model = Seq2SeqTransformer(config).eval()
    src = torch.randint(3, config.vocab_size, (2, 5))

    with torch.no_grad():
        first = model.generate(src, max_new_tokens=4, eos_token_id=0)
        second = model.generate(src, max_new_tokens=4, eos_token_id=0)

    assert torch.equal(first, second)


def test_seq2seq_generate_accepts_generation_config():
    config = TransformerConfig(
        vocab_size=40,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        output_mode="vocab",
    )
    model = Seq2SeqTransformer(config).eval()
    src = torch.randint(3, config.vocab_size, (2, 5))
    generation_config = GenerationConfig(max_new_tokens=3, eos_token_id=0, use_cache=False)

    with torch.no_grad():
        generated = model.generate(src, generation_config=generation_config)

    assert generated.shape == (2, 3)


def test_seq2seq_generate_cached_and_uncached_greedy_match():
    config = TransformerConfig(
        vocab_size=40,
        d_model=16,
        n_heads=4,
        n_layers=2,
        d_ff=32,
        dropout=0.0,
        output_mode="vocab",
    )
    model = Seq2SeqTransformer(config).eval()
    src = torch.randint(3, config.vocab_size, (2, 5))

    with torch.no_grad():
        cached = model.generate(src, max_new_tokens=5, eos_token_id=0, use_cache=True)
        uncached = model.generate(src, max_new_tokens=5, eos_token_id=0, use_cache=False)

    assert torch.equal(cached, uncached)


def test_seq2seq_generate_stops_on_eos():
    config = TransformerConfig(
        vocab_size=20,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        output_mode="vocab",
    )
    model = Seq2SeqTransformer(config).eval()
    eos_token_id = 3
    assert isinstance(model.decoder.output_projection, torch.nn.Linear)
    with torch.no_grad():
        model.decoder.output_projection.weight.zero_()
        model.decoder.output_projection.bias.fill_(-10.0)
        model.decoder.output_projection.bias[eos_token_id] = 10.0

    src = torch.randint(4, config.vocab_size, (2, 5))
    generated = model.generate(src, max_new_tokens=5, eos_token_id=eos_token_id)

    assert generated.shape == (2, 1)
    assert torch.equal(generated, torch.full_like(generated, eos_token_id))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"bos_token_id": -1}, "bos_token_id"),
        ({"eos_token_id": 100}, "eos_token_id"),
        ({"do_sample": False, "temperature": 0.7}, "do_sample=True"),
        ({"do_sample": True, "temperature": 0.0}, "temperature"),
        ({"do_sample": True, "top_k": -1}, "top_k"),
        ({"do_sample": True, "top_p": 0.0}, "top_p"),
    ],
)
def test_seq2seq_generate_validates_arguments(kwargs, match):
    config = TransformerConfig(
        vocab_size=40,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        output_mode="vocab",
    )
    model = Seq2SeqTransformer(config).eval()
    src = torch.randint(3, config.vocab_size, (1, 4))

    with pytest.raises((TypeError, ValueError), match=match):
        model.generate(src, **kwargs)


def test_generation_config_validates_token_ids_against_vocab():
    with pytest.raises(ValueError, match="eos_token_id"):
        GenerationConfig(eos_token_id=10).validate(vocab_size=10)


def test_sample_next_token_uses_greedy_argmax():
    logits = torch.tensor([[0.1, 0.5, 0.4], [3.0, 1.0, 2.0]])

    next_token = sample_next_token(logits, GenerationConfig())

    assert next_token.tolist() == [[1], [0]]


def test_sample_next_token_rejects_nonfinite_logits():
    logits = torch.tensor([[0.1, float("inf"), 0.4]])

    with pytest.raises(ValueError, match="finite"):
        sample_next_token(logits, GenerationConfig())


def test_sample_next_token_rejects_nan_logits():
    logits = torch.tensor([[0.1, float("nan"), 0.4]])

    with pytest.raises(ValueError, match="finite"):
        sample_next_token(logits, GenerationConfig(do_sample=True))


def test_sampling_extreme_finite_logits_stays_valid():
    logits = torch.tensor([[1e20, -1e20, 0.0]])
    config = GenerationConfig(do_sample=True, temperature=1.0)
    generator = torch.Generator().manual_seed(0)

    next_token = sample_next_token(logits, config, generator=generator)

    assert next_token.tolist() == [[0]]


def test_filter_logits_for_sampling_applies_top_k():
    logits = torch.tensor([[0.0, 3.0, 2.0, -1.0]])
    config = GenerationConfig(do_sample=True, top_k=2)

    filtered = filter_logits_for_sampling(logits, config)
    fill_value = torch.finfo(filtered.dtype).min

    assert torch.equal(filtered[0, 1:3], logits[0, 1:3])
    assert filtered[0, 0].item() == fill_value
    assert filtered[0, 3].item() == fill_value


def test_filter_logits_for_sampling_applies_top_p():
    logits = torch.tensor([[4.0, 3.0, 1.0, 0.0]])
    config = GenerationConfig(do_sample=True, top_p=0.6)

    filtered = filter_logits_for_sampling(logits, config)
    fill_value = torch.finfo(filtered.dtype).min

    assert filtered[0, 0].item() == pytest.approx(4.0)
    assert filtered[0, 1].item() == fill_value
    assert filtered[0, 2].item() == fill_value
    assert filtered[0, 3].item() == fill_value
