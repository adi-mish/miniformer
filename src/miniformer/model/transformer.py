import torch
import torch.nn as nn
import os
from typing import Dict, List, Optional, Union, Tuple

from miniformer.model.attention import MultiHeadAttention
from miniformer.model.embedding import TokenEmbedding, PositionalEncoding
from miniformer.model.feedforward import FeedForward
from miniformer.config.model_config import TransformerConfig


class EncoderLayer(nn.Module):
    """Transformer encoder layer"""
    
    def __init__(self, config: TransformerConfig):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        self.feed_forward = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, 1, 1, seq_len)
            
        Returns:
            output: Transformed tensor (batch_size, seq_len, d_model)
            attention: Attention weights
        """
        # Self-attention block with residual connection and layer norm
        attn_output, attention = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward block with residual connection and layer norm
        ff_output = self.feed_forward(x)
        output = self.norm2(x + self.dropout(ff_output))
        
        return output, attention


class Encoder(nn.Module):
    """Transformer encoder"""
    
    def __init__(self, config: TransformerConfig):
        """
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Create token embedding only if input_dim is not specified
        if config.input_dim is None:
            self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        else:
            self.token_embedding = None
            
        self.position_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )
        self.layers = nn.ModuleList([
            EncoderLayer(config)
            for _ in range(config.n_layers)
        ])
        self.attention_weights = None
        
        # Apply weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input token ids (batch_size, seq_len) or feature vectors (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, 1, 1, seq_len)
            
        Returns:
            output: Encoded representation (batch_size, seq_len, d_model)
        """
        # For token IDs, use token embedding; otherwise, use input features directly
        if self.token_embedding is not None:
            # Input is token IDs
            if x.dim() == 2:
                x = self.token_embedding(x)
            else:
                raise ValueError("When token_embedding is used, input should be 2D tensor of token IDs")
        else:
            # Input is already embedded features
            if x.dim() != 3 or x.size(-1) != self.config.d_model:
                raise ValueError(f"Expected input features of shape (batch_size, seq_len, {self.config.d_model})")
        
        # Add positional encoding
        x = self.position_encoding(x)
        
        # Store attention weights for visualization
        self.attention_weights = []
        
        # Apply encoder layers
        for layer in self.layers:
            x, attention = layer(x, mask)
            self.attention_weights.append(attention)
            
        return x


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
