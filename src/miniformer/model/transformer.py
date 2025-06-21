import torch
import torch.nn as nn
import os
from typing import Dict, List, Optional, Union, Tuple

from miniformer.model.attention import MultiHeadAttention
from miniformer.model.embedding import TokenEmbedding, PositionalEncoding
from miniformer.model.feedforward import FeedForward
from miniformer.config.model_config import TransformerConfig

from miniformer.model.encoder import Encoder

class Transformer(nn.Module):
    """Transformer model for sequence classification or prediction tasks"""
    
    def __init__(self, config: Optional[TransformerConfig] = None, **kwargs):
        """
        Args:
            config: Model configuration
            **kwargs: Override configuration parameters
        """
        super().__init__()
        
        # Create config if not provided
        if config is None:
            if kwargs:
                config = TransformerConfig(**kwargs)
            else:
                config = TransformerConfig()
        elif kwargs:
            # Update config with any provided kwargs
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                    
        self.config = config
        self.encoder = Encoder(config)
        
        # Use output_dim for final projection
        if config.output_dim is None:
            raise ValueError("output_dim must be specified in TransformerConfig for the output layer.")
        self.output_layer = nn.Linear(config.d_model, config.output_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input token ids (batch_size, seq_len) or feature vectors (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, 1, 1, seq_len)
            
        Returns:
            output: Output logits (batch_size, seq_len, output_dim)
        """
        # Create mask if not provided
        if mask is None and x is not None:
            # For token IDs (2D tensor)
            if x.dim() == 2:
                mask = (x != 0).unsqueeze(1).unsqueeze(2)
            # For feature vectors (3D tensor), assume all positions are valid
            else:
                mask = torch.ones((x.size(0), 1, 1, x.size(1)), device=x.device).bool()
            
        # Encode input
        encoded = self.encoder(x, mask)
        
        # Project to output dimension
        output = self.output_layer(encoded)
        
        return output
    
    def _create_mask(self, x):
        """Create a mask to hide padding tokens"""
        # x: (batch_size, seq_len)
        mask = (x != 0).unsqueeze(1).unsqueeze(2)
        return mask
    
    def get_attention_weights(self, x):
        """Get attention weights for visualization"""
        mask = self._create_mask(x)
        _ = self.forward(x, mask)
        return self.encoder.attention_weights
        
    def save_pretrained(self, save_dir: str) -> None:
        """Save model and configuration to directory"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(save_dir, "model.pt")
        torch.save(self.state_dict(), model_path)
        
        # Save configuration
        config_path = os.path.join(save_dir, "config.json")
        self.config.save_json(config_path)
        
    @classmethod
    def from_pretrained(cls, model_dir: str) -> "Transformer":
        """Load model from directory"""
        # Load configuration
        config_path = os.path.join(model_dir, "config.json")
        config = TransformerConfig.from_json(config_path)
        
        # Create model with loaded config
        model = cls(config)
        
        # Load weights
        model_path = os.path.join(model_dir, "model.pt")
        model.load_state_dict(torch.load(model_path))
        
        return model
