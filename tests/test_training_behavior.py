import torch
import pytest
from miniformer.model.transformer import Transformer, TransformerConfig


def test_mini_training_loop_convergence():
    """Test that model can overfit a tiny dataset (convergence smoke test)."""
    config = TransformerConfig(
        vocab_size=20,
        d_model=32,
        n_heads=4,
        n_layers=2,
        output_dim=20,
        dropout=0.0  # Disable dropout for consistent training
    )
    model = Transformer(config)
    
    # Create tiny dataset - simple copy task
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = x.clone()  # Copy task: predict the same sequence
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model.train()
    losses = []
    
    for step in range(100):  # Enough steps to see convergence
        optimizer.zero_grad()
        
        logits = model(x)  # [batch, seq, vocab_size]
        loss = loss_fn(logits.view(-1, 20), y.view(-1))
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Early stopping if loss is very low
        if loss.item() < 0.01:
            break
    
    # Loss should decrease significantly
    initial_loss = losses[0]
    final_loss = losses[-1]
    
    assert final_loss < initial_loss * 0.1, \
        f"Model failed to converge: initial={initial_loss:.4f}, final={final_loss:.4f}"
    
    # Should reach reasonable accuracy on this simple task
    assert final_loss < 1.0, f"Final loss too high: {final_loss:.4f}"


def test_gradient_accumulation_equivalence():
    """Test that gradient accumulation produces equivalent results to larger batches."""
    config = TransformerConfig(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        n_layers=2,
        output_dim=50
    )
    
    # Create two identical models
    model1 = Transformer(config)
    model2 = Transformer(config)
    
    # Sync their weights
    model2.load_state_dict(model1.state_dict())
    
    # Create data
    x = torch.randint(0, 50, (8, 6))  # Batch size 8
    y = torch.randint(0, 50, (8, 6))
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Method 1: Regular batch training
    model1.train()
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.1)
    
    optimizer1.zero_grad()
    logits1 = model1(x)
    loss1 = loss_fn(logits1.view(-1, 50), y.view(-1))
    loss1.backward()
    optimizer1.step()
    
    # Method 2: Gradient accumulation (2 steps of batch size 4)
    model2.train()
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    
    optimizer2.zero_grad()
    
    # First half
    logits2a = model2(x[:4])
    loss2a = loss_fn(logits2a.view(-1, 50), y[:4].view(-1))
    (loss2a / 2).backward()  # Scale by accumulation steps
    
    # Second half
    logits2b = model2(x[4:])
    loss2b = loss_fn(logits2b.view(-1, 50), y[4:].view(-1))
    (loss2b / 2).backward()  # Scale by accumulation steps
    
    optimizer2.step()
    
    # Models should have similar parameters after equivalent updates
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-5), \
            "Gradient accumulation should produce equivalent results"


def test_learning_rate_sensitivity():
    """Test that model is sensitive to learning rate changes."""
    config = TransformerConfig(
        vocab_size=30,
        d_model=32,
        n_heads=4,
        n_layers=1,
        output_dim=30
    )
    
    # Test with different learning rates
    learning_rates = [0.001, 0.01, 0.1]
    final_losses = {}
    
    x = torch.tensor([[1, 2, 3, 4, 5]])
    y = torch.tensor([[2, 3, 4, 5, 6]])  # Next token prediction
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for lr in learning_rates:
        model = Transformer(config)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        # Initialize loss to avoid unbound error
        loss = torch.tensor(0.0)
        
        # Train for a few steps
        for _ in range(20):
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, 30), y.view(-1))
            loss.backward()
            optimizer.step()
        
        final_losses[lr] = loss.item()
    
    # Different learning rates should produce different final losses
    losses_list = list(final_losses.values())
    assert not all(abs(losses_list[0] - loss) < 1e-3 for loss in losses_list[1:]), \
        "Model should be sensitive to learning rate changes"


def test_weight_decay_effect():
    """Test that weight decay affects model parameters."""
    config = TransformerConfig(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        n_layers=1
    )
    
    # Create two identical models
    model_no_decay = Transformer(config)
    model_with_decay = Transformer(config)
    
    # Sync their weights
    model_with_decay.load_state_dict(model_no_decay.state_dict())
    
    # Different optimizers
    optimizer_no_decay = torch.optim.Adam(model_no_decay.parameters(), lr=0.01, weight_decay=0.0)
    optimizer_with_decay = torch.optim.Adam(model_with_decay.parameters(), lr=0.01, weight_decay=0.01)
    
    x = torch.randint(0, 50, (4, 8))
    
    # Train both models
    for _ in range(50):
        # No decay model
        optimizer_no_decay.zero_grad()
        output_no_decay = model_no_decay(x)
        loss_no_decay = output_no_decay.sum()
        loss_no_decay.backward()
        optimizer_no_decay.step()
        
        # With decay model
        optimizer_with_decay.zero_grad()
        output_with_decay = model_with_decay(x)
        loss_with_decay = output_with_decay.sum()
        loss_with_decay.backward()
        optimizer_with_decay.step()
    
    # Models should have different parameters due to weight decay
    param_diffs = []
    for p1, p2 in zip(model_no_decay.parameters(), model_with_decay.parameters()):
        param_diffs.append(torch.norm(p1 - p2).item())
    
    avg_diff = sum(param_diffs) / len(param_diffs)
    assert avg_diff > 1e-4, f"Weight decay should cause parameter differences (avg_diff={avg_diff})"