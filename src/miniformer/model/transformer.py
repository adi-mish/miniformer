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

        # Create or update config
        if config is None:
            if kwargs:
                config = TransformerConfig(**kwargs)
            else:
                config = TransformerConfig()
        elif kwargs:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config

        # ✱ No local input_projection here (Encoder already handles input_dim→d_model) ✱
        self.input_projection = None

        self.encoder = Encoder(config)

        # Task head: linear only if output_dim requested, else identity
        if config.output_dim is not None:
            self.output_layer = nn.Linear(config.d_model, config.output_dim)
        else:
            self.output_layer = nn.Identity()

    def forward(self, x, mask=None):
        """
        Args:
            x: Input token ids (batch_size, seq_len) or feature vectors (batch_size, seq_len, input_dim)
            mask: Attention mask (batch_size, 1, 1, seq_len)

        Returns:
            output: Output logits or embeddings
                    – if output_dim was set: (batch_size, seq_len, output_dim)
                    – otherwise:       (batch_size, seq_len, d_model)
        """
        # Build mask if not provided (padding + causal for token inputs)
        if mask is None and x is not None:
            B, S = x.shape[0], x.shape[1]
            if x.dim() == 2:
                pad_mask = (x != 0).unsqueeze(1).unsqueeze(2)                  # [B,1,1,S]
                causal    = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
                causal_mask = causal.unsqueeze(0).unsqueeze(0)                # [1,1,S,S]
                mask = pad_mask & causal_mask                                 # [B,1,S,S]
            else:
                mask = torch.ones((B, 1, 1, S), device=x.device, dtype=torch.bool)

        # Encode to [B, S, d_model]
        encoded = self.encoder(x, mask)

        # Final projection or identity
        output = self.output_layer(encoded)
        return output

    def _create_mask(self, x):
        """Create a mask to hide padding tokens"""
        return (x != 0).unsqueeze(1).unsqueeze(2)

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
        config_path = os.path.join(model_dir, "config.json")
        config = TransformerConfig.from_json(config_path)

        model = cls(config)
        model_path = os.path.join(model_dir, "model.pt")
        model.load_state_dict(torch.load(model_path))
        return model
