import torch
import pytest
import tempfile
import os
from miniformer.model.transformer import Transformer, TransformerConfig


def test_save_load_round_trip():
    """Test that model can be saved and loaded with identical behavior."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2
    )
    model = Transformer(config)
    
    # Create test input
    x = torch.randint(0, 100, (2, 10))
    
    model.eval()
    with torch.no_grad():
        original_output = model(x)
    
    # Save to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        model.save_pretrained(temp_dir)
        
        # Load model
        loaded_model = Transformer.from_pretrained(temp_dir)
        loaded_model.eval()
        
        with torch.no_grad():
            loaded_output = loaded_model(x)
        
        # Outputs should be identical
        assert torch.allclose(original_output, loaded_output, atol=1e-6)
        
        # State dicts should match
        original_state = model.state_dict()
        loaded_state = loaded_model.state_dict()
        
        assert original_state.keys() == loaded_state.keys()
        for key in original_state.keys():
            assert torch.allclose(original_state[key], loaded_state[key], atol=1e-6)


def test_config_serialization():
    """Test that TransformerConfig can be saved and loaded."""
    config = TransformerConfig(
        vocab_size=1000,
        d_model=128,
        n_heads=8,
        n_layers=6,
        d_ff=512,
        dropout=0.15,
        activation="swiglu"
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        
        # Save config using json serialization
        import json
        config_dict = {
            'vocab_size': config.vocab_size,
            'd_model': config.d_model,
            'n_heads': config.n_heads,
            'n_layers': config.n_layers,
            'd_ff': config.d_ff,
            'dropout': config.dropout,
            'activation': config.activation
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f)
        
        # Load config
        with open(config_path, 'r') as f:
            loaded_dict = json.load(f)
        
        loaded_config = TransformerConfig(**loaded_dict)
        
        # All attributes should match
        assert loaded_config.vocab_size == config.vocab_size
        assert loaded_config.d_model == config.d_model
        assert loaded_config.n_heads == config.n_heads
        assert loaded_config.n_layers == config.n_layers
        assert loaded_config.d_ff == config.d_ff
        assert loaded_config.dropout == config.dropout
        assert loaded_config.activation == config.activation
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_device_transfer():
    """Test that model works correctly after moving to CUDA."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2
    )
    model = Transformer(config)
    
    # Test on CPU first
    x_cpu = torch.randint(0, 100, (2, 10))
    model.eval()
    with torch.no_grad():
        output_cpu = model(x_cpu)
    
    # Move to CUDA
    model = model.cuda()
    x_cuda = x_cpu.cuda()
    
    with torch.no_grad():
        output_cuda = model(x_cuda)
    
    # Results should be equivalent (allowing for small numerical differences)
    assert torch.allclose(output_cpu, output_cuda.cpu(), atol=1e-5)


@pytest.mark.skipif(not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available(), 
                    reason="MPS not available")
def test_mps_device_transfer():
    """Test that model works correctly after moving to MPS (Apple Silicon)."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2
    )
    model = Transformer(config)
    
    # Test on CPU first
    x_cpu = torch.randint(0, 100, (2, 10))
    model.eval()
    with torch.no_grad():
        output_cpu = model(x_cpu)
    
    # Move to MPS
    model = model.to('mps')
    x_mps = x_cpu.to('mps')
    
    with torch.no_grad():
        output_mps = model(x_mps)
    
    # Results should be equivalent (allowing for small numerical differences)
    assert torch.allclose(output_cpu, output_mps.cpu(), atol=1e-5)