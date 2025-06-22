import torch
import pytest
from miniformer.model.transformer import Transformer, TransformerConfig


def test_backward_pass_finite_gradients():
    """Test that backward pass produces finite gradients."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2,
        output_dim=10  # Classification task
    )
    model = Transformer(config)
    
    # Create dummy data
    x = torch.randint(0, 100, (2, 8))
    targets = torch.randint(0, 10, (2, 8))
    
    # Forward pass
    logits = model(x)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, 10), targets.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Check all gradients are finite
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"


def test_dropout_train_vs_eval():
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
    with torch.no_grad():
        output3 = model(x)
        output4 = model(x)
    
    # Should be different (with high probability given 50% dropout)
    assert not torch.allclose(output3, output4, atol=1e-6), "Train mode should have stochastic dropout"


def test_mini_fit_toy_task():
    """Test that model can overfit a tiny dataset (sanity check for learning)."""
    config = TransformerConfig(
        vocab_size=10,
        d_model=32,
        n_heads=4,
        n_layers=2,
        output_dim=10
    )
    model = Transformer(config)
    
    # Create tiny dataset - simple copy task
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = x.clone()  # Copy task
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model.train()
    initial_loss = None
    final_loss = None
    
    for step in range(50):  # Small number of steps
        optimizer.zero_grad()
        
        logits = model(x)
        loss = loss_fn(logits.view(-1, 10), y.view(-1))
        
        if step == 0:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        final_loss = loss.item()
    
    # Loss should decrease significantly (model should overfit small dataset)
    assert initial_loss is not None, "Initial loss was not recorded"
    assert final_loss is not None, "Final loss was not recorded"
    assert final_loss < initial_loss * 0.5, f"Loss should decrease: {initial_loss} -> {final_loss}"


def test_gradient_flow():
    """Test that gradients flow through all layers."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=3  # Multiple layers to test gradient flow
    )
    model = Transformer(config)
    
    x = torch.randint(0, 100, (2, 10))
    
    # Forward pass
    output = model(x)
    loss = output.sum()  # Simple loss for gradient computation
    
    # Backward pass
    loss.backward()
    
    # Check that all layers have gradients
    layers_with_grads = 0
    total_layers = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_layers += 1
            if param.grad is not None and param.grad.abs().sum() > 1e-8:
                layers_with_grads += 1
    
    # Most parameters should have non-zero gradients
    gradient_ratio = layers_with_grads / total_layers
    assert gradient_ratio > 0.8, f"Only {gradient_ratio:.2%} of parameters have gradients"


def test_weight_initialization():
    """Test that weights are properly initialized (not all zeros or ones)."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2
    )
    model = Transformer(config)
    
    for name, param in model.named_parameters():
        if param.dim() >= 2:  # Skip biases and 1D parameters
            # Weights shouldn't be all zeros
            assert not torch.allclose(param, torch.zeros_like(param)), f"{name} is all zeros"
            
            # Weights shouldn't be all the same value
            assert param.std() > 1e-6, f"{name} has zero variance"
            
            # Weights should be reasonable scale (not too large or small)
            assert param.abs().max() < 10.0, f"{name} has very large values"
            assert param.abs().mean() > 1e-4, f"{name} has very small values"