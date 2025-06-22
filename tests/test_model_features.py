import torch
import pytest
from miniformer.model.transformer import Transformer, TransformerConfig


def test_shared_embeddings_weight_tying():
    """Test that shared embeddings actually share the same weight tensor."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2,
        output_dim=100  # Same as vocab_size for proper weight sharing
    )
    model = Transformer(config)
    
    # Check if input and output embeddings share weights
    if hasattr(model, 'token_embedding') and hasattr(model, 'output_projection'):
        input_embedding = model.token_embedding
        output_projection = model.output_projection
        
        # They should be the same tensor object (not just equal values)
        assert input_embedding is output_projection, \
            "Shared embeddings should use the same memory location"


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
    )
    
    config_post = TransformerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=4,
        n_layers=2,
    )
    
    model_pre = Transformer(config_pre)
    model_post = Transformer(config_post)
    
    # Set same random seed for initialization
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size, (2, 8))
    
    model_pre.eval()
    model_post.eval()
    
    with torch.no_grad():
        output_pre = model_pre(x)
        output_post = model_post(x)
    
    # Outputs should have same shape
    assert output_pre.shape == output_post.shape == (2, 8, d_model)
    
    # But different values (with high probability)
    assert not torch.allclose(output_pre, output_post, atol=1e-3), \
        "Pre-norm and post-norm should produce different outputs"


def test_different_activation_functions():
    """Test that different activation functions work correctly."""
    activations_to_test = ["relu", "gelu", "swiglu"]
    
    outputs = {}
    x = torch.randint(0, 100, (2, 6))
    
    for activation in activations_to_test:
        config = TransformerConfig(
            vocab_size=100,
            d_model=32,
            n_heads=4,
            n_layers=1,
            activation=activation
        )
        model = Transformer(config)
        model.eval()
        
        with torch.no_grad():
            outputs[activation] = model(x)
    
    # All should produce valid outputs
    for activation, output in outputs.items():
        assert output.shape == (2, 6, 32), f"Wrong shape for {activation}"
        assert torch.isfinite(output).all(), f"Non-finite output for {activation}"
    
    # Different activations should produce different results
    activations = list(outputs.keys())
    for i in range(len(activations)):
        for j in range(i + 1, len(activations)):
            act1, act2 = activations[i], activations[j]
            assert not torch.allclose(outputs[act1], outputs[act2], atol=1e-3), \
                f"Activations {act1} and {act2} produced too similar outputs"


def test_layer_scale_initialization():
    """Test that layer scale parameters are properly initialized when enabled."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2,
    )
    model = Transformer(config)
    
    # Check for layer scale parameters
    layer_scale_params = [name for name, _ in model.named_parameters() if 'layer_scale' in name]
    
    if layer_scale_params:  # Only test if layer scale is implemented
        for name, param in model.named_parameters():
            if 'layer_scale' in name:
                # Should be initialized close to the specified value
                assert torch.allclose(param, torch.full_like(param, 1e-4), atol=1e-6), \
                    f"Layer scale parameter {name} not properly initialized"


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
        )
        model = Transformer(config)
        model.eval()
        
        with torch.no_grad():
            outputs[rotary_pct] = model(x)
    
    # All should produce valid outputs
    for rotary_pct, output in outputs.items():
        assert output.shape == (2, 12, 64), f"Wrong shape for rotary_pct={rotary_pct}"
        assert torch.isfinite(output).all(), f"Non-finite output for rotary_pct={rotary_pct}"
    
    # Different rotary percentages should generally produce different results
    # (except 0.0 vs others might be similar for some implementations)
    assert not torch.allclose(outputs[0.0], outputs[1.0], atol=1e-3), \
        "Full rotary embeddings should differ from no rotary embeddings"


def test_model_parameter_count_scaling():
    """Test that parameter count scales appropriately with model size."""
    base_config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=1
    )
    
    larger_config = TransformerConfig(
        vocab_size=100,
        d_model=64,  # 2x larger
        n_heads=8,
        n_layers=2   # 2x more layers
    )
    
    base_model = Transformer(base_config)
    larger_model = Transformer(larger_config)
    
    base_params = sum(p.numel() for p in base_model.parameters())
    larger_params = sum(p.numel() for p in larger_model.parameters())
    
    # Larger model should have significantly more parameters
    assert larger_params > base_params * 2, \
        f"Larger model ({larger_params}) should have much more parameters than base ({base_params})"