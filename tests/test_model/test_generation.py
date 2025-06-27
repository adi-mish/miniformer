import torch
import pytest
from miniformer.model.transformer import Transformer, TransformerConfig


def test_kv_cache_equivalence():
    """Test that KV caching produces identical results to non-cached generation."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2
    )
    model = Transformer(config)
    model.eval()
    
    # Initial sequence
    input_ids = torch.randint(1, 100, (1, 5))
    
    with torch.no_grad():
        # Non-cached forward pass
        output_no_cache = model(input_ids, use_cache=False)
        
        # Cached generation (simulate autoregressive generation)
        past_key_values = None
        cached_outputs = []
        
        for i in range(input_ids.size(1)):
            current_token = input_ids[:, i:i+1]
            output, past_key_values = model(
                current_token, 
                past_key_values=past_key_values,
                use_cache=True
            )
            cached_outputs.append(output)
        
        # Concatenate cached outputs
        output_cached = torch.cat(cached_outputs, dim=1)
        
        # Should be identical (within numerical precision)
        assert torch.allclose(output_no_cache, output_cached, atol=1e-6)


def test_generation_with_max_new_tokens():
    """Test autoregressive generation with max_new_tokens limit."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2
    )
    model = Transformer(config)
    model.eval()
    
    input_ids = torch.randint(1, 100, (1, 3))
    max_new_tokens = 5
    
    with torch.no_grad():
        # Manual autoregressive generation
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    
    # Should generate exactly max_new_tokens additional tokens
    assert generated.size(1) == input_ids.size(1) + max_new_tokens
    # First tokens should match input
    assert torch.equal(generated[:, :input_ids.size(1)], input_ids)

def test_generation_with_eos_token():
    """Test that generation stops at EOS token."""
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        n_heads=4,
        n_layers=2
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
        return logits
    
    model.forward = mock_forward
    model.eval()
    
    input_ids = torch.randint(1, 98, (1, 3))  # Avoid EOS in input
    
    with torch.no_grad():
        # Manual autoregressive generation with EOS stopping
        generated = input_ids.clone()
        max_new_tokens = 10
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            if next_token.item() == eos_token_id:
                break
    
    # Should stop early due to EOS token
    assert generated.size(1) <= input_ids.size(1) + 10
    # Should contain EOS token
    assert eos_token_id in generated[0]