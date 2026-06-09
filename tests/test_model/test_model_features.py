import pytest
import torch
import torch.nn as nn

from miniformer.model.seq2seq_transformer import Seq2SeqTransformer
from miniformer.model.transformer import Transformer, TransformerConfig


def test_shared_embeddings_weight_tying():
    """Test that shared embeddings actually share the same weight tensor."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2,
        output_mode="vocab",
    )
    model = Transformer(config)

    assert model.token_embedding is not None
    assert isinstance(model.output_projection, nn.Identity)
    assert model._tied_weights is True

    x = torch.randint(0, config.vocab_size, (2, 4))
    logits = model(x).output
    loss = logits.sum()
    loss.backward()
    assert model.token_embedding.weight.grad is not None


def test_pre_norm_vs_post_norm():
    """Test that pre-norm and post-norm configurations produce different behaviors."""
    vocab_size = 100
    d_model = 32

    # Create identical configs except for normalization order
    config_pre = TransformerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        pre_norm=True,
    )

    config_post = TransformerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        pre_norm=False,
    )

    model_pre = Transformer(config_pre)
    model_post = Transformer(config_post)

    # Set same random seed for initialization
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size, (2, 8))

    model_pre.eval()
    model_post.eval()

    with torch.no_grad():
        output_pre = model_pre(x).output
        output_post = model_post(x).output

    # Outputs should have same shape
    assert output_pre.shape == output_post.shape == (2, 8, d_model)

    # But different values (with high probability)
    assert not torch.allclose(
        output_pre, output_post, atol=1e-3
    ), "Pre-norm and post-norm should produce different outputs"


def test_different_activation_functions():
    """Test that different activation functions work correctly."""
    activations_to_test = ["relu", "gelu", "swiglu"]

    outputs = {}
    x = torch.randint(0, 100, (2, 6))

    for activation in activations_to_test:
        config = TransformerConfig(
            vocab_size=100, d_model=32, n_heads=4, n_layers=1, activation=activation
        )
        model = Transformer(config)
        model.eval()

        with torch.no_grad():
            outputs[activation] = model(x).output

    # All should produce valid outputs
    for activation, output in outputs.items():
        assert output.shape == (2, 6, 32), f"Wrong shape for {activation}"
        assert torch.isfinite(output).all(), f"Non-finite output for {activation}"

    # Different activations should produce different results
    activations = list(outputs.keys())
    for i in range(len(activations)):
        for j in range(i + 1, len(activations)):
            act1, act2 = activations[i], activations[j]
            assert not torch.allclose(
                outputs[act1], outputs[act2], atol=1e-3
            ), f"Activations {act1} and {act2} produced too similar outputs"


def test_rotary_embedding_implementation():
    """Test that rotary embeddings work with different percentages."""
    x = torch.randint(0, 100, (2, 12))

    outputs = {}

    for rotary_pct in [0.0, 0.25, 0.5, 1.0]:
        config = TransformerConfig(
            vocab_size=100,
            d_model=64,  # Use 64 to ensure divisibility
            n_heads=8,
            n_layers=1,
            position_mode="learned" if rotary_pct == 0.0 else "learned+rope",
            rotary_pct=rotary_pct,
        )
        model = Transformer(config)
        model.eval()

        with torch.no_grad():
            outputs[rotary_pct] = model(x).output

    # All should produce valid outputs
    for rotary_pct, output in outputs.items():
        assert output.shape == (2, 12, 64), f"Wrong shape for rotary_pct={rotary_pct}"
        assert torch.isfinite(output).all(), f"Non-finite output for rotary_pct={rotary_pct}"

    # Different rotary percentages should generally produce different results
    # (except 0.0 vs others might be similar for some implementations)
    assert not torch.allclose(
        outputs[0.0], outputs[1.0], atol=1e-3
    ), "Full rotary embeddings should differ from no rotary embeddings"


@pytest.mark.parametrize(
    "position_mode,rotary_pct,has_learned_positions,has_rope",
    [
        ("learned", 0.0, True, False),
        ("rope", 0.5, False, True),
        ("learned+rope", 0.5, True, True),
    ],
)
def test_position_mode_controls_learned_embeddings_and_rope(
    position_mode, rotary_pct, has_learned_positions, has_rope
):
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=1,
        position_mode=position_mode,
        rotary_pct=rotary_pct,
    )
    encoder_model = Transformer(config)
    seq2seq_model = Seq2SeqTransformer(config)

    assert (encoder_model.encoder.pos_embedding is not None) is has_learned_positions
    assert (seq2seq_model.decoder.position_encoding is not None) is has_learned_positions
    assert (encoder_model.encoder.layers[0].self_attention.rotary_dim > 0) is has_rope
    assert (seq2seq_model.decoder.layers[0].self_attention.rotary_dim > 0) is has_rope


def test_initializer_range_controls_projection_head_and_backbone():
    torch.manual_seed(0)
    config = TransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=8,
        n_layers=1,
        output_mode="projection",
        output_dim=7,
        initializer_range=0.003,
    )
    model = Transformer(config)

    backbone_std = model.encoder.layers[0].self_attention.wq.weight.std().item()
    head = model.output_projection
    assert isinstance(head, nn.Linear)
    head_std = head.weight.std().item()

    assert 0.0 < backbone_std < 0.006
    assert 0.0 < head_std < 0.006
    assert torch.equal(head.bias, torch.zeros_like(head.bias))


def test_seq2seq_uses_config_initializer_without_outer_reinitialization():
    torch.manual_seed(0)
    config = TransformerConfig(
        vocab_size=100,
        d_model=64,
        n_heads=8,
        n_layers=1,
        output_mode="vocab",
        initializer_range=0.003,
    )
    model = Seq2SeqTransformer(config)

    assert model.encoder.token_embedding is not None
    assert model.encoder.token_embedding.weight.std().item() < 0.006
    assert isinstance(model.decoder.output_projection, nn.Linear)
    assert model.decoder.output_projection.weight.std().item() < 0.006


def test_model_parameter_count_scaling():
    """Test that parameter count scales appropriately with model size."""
    base_config = TransformerConfig(vocab_size=100, d_model=32, n_heads=4, n_layers=1)

    larger_config = TransformerConfig(
        vocab_size=100, d_model=64, n_heads=8, n_layers=2  # 2x larger  # 2x more layers
    )

    base_model = Transformer(base_config)
    larger_model = Transformer(larger_config)

    base_params = sum(p.numel() for p in base_model.parameters())
    larger_params = sum(p.numel() for p in larger_model.parameters())

    # Larger model should have significantly more parameters
    assert (
        larger_params > base_params * 2
    ), f"Larger model ({larger_params}) should have much more parameters than base ({base_params})"
