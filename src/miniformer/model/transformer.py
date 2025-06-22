import torch
import torch.nn as nn
import os
from typing import Optional

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

        # output_layer is no longer used in forward, but we keep it if needed elsewhere
        if config.output_dim is None:
            raise ValueError("output_dim must be specified in TransformerConfig for the output layer.")
        self.output_layer = nn.Linear(config.d_model, config.output_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input token ids (batch_size, seq_len) or feature vectors (batch_size, seq_len, d_model)
            mask: Attention mask (batch_size, 1, 1, seq_len)

        Returns:
            encoded: Raw d_model embeddings (batch_size, seq_len, d_model)
        """
        # Build mask if not provided
        if mask is None and x is not None:
            B, S = x.shape[0], x.shape[1]
            if x.dim() == 2:
                # padding mask: True where x != 0
                pad_mask = (x != 0).unsqueeze(1).unsqueeze(2)                  # [B,1,1,S]
                # causal mask: lower triangular [S,S]
                causal = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
                causal_mask = causal.unsqueeze(0).unsqueeze(0)                # [1,1,S,S]
                mask = pad_mask & causal_mask                                 # broadcast→[B,1,S,S]
            else:
                # no padding or causality needed for feature inputs
                mask = torch.ones((B, 1, 1, S), device=x.device, dtype=torch.bool)

        # Run the encoder with the combined mask → [B, S, d_model]
        encoded = self.encoder(x, mask)

        # Return raw d_model embeddings (no vocab projection)
        return encoded

    def _create_mask(self, x):
        """Create a mask to hide padding tokens"""
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
