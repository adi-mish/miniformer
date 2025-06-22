from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import json
import os
from pathlib import Path


@dataclass
class TransformerConfig:
    """Configuration class for Transformer models"""
    
    # Model architecture
    vocab_size: int = 10000
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    activation: str = "gelu"  # "gelu", "relu", or "swiglu"
    layer_norm_eps: float = 1e-5
    max_seq_len: int = 1024
    
    # Input/Output dimensions
    input_dim: Optional[int] = None  # If provided, model accepts feature vectors directly
    output_dim: Optional[int] = None  # If provided, model projects to this dimension instead of vocab_size
    
    # Training parameters
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 0
    lr_scheduler: str = "linear"  # "linear", "cosine", or "constant"
    batch_size: int = 32
    
    # Other parameters
    initializer_range: float = 0.02
    model_name: str = "miniformer-base"
    
    def __post_init__(self):
        """Validate configuration and set defaults"""
        # ✱ Validate d_model/n_heads ✱
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")

        # ✱ Validate activation ✱
        valid_activations = {"gelu", "relu", "swiglu"}
        if self.activation.lower() not in valid_activations:
            raise ValueError(f"Unknown activation: {self.activation}")

        if self.input_dim is not None and self.input_dim != self.d_model and self.output_dim is None:
            raise ValueError(
                "input_dim differs from d_model – please set output_dim explicitly "
                "or keep input_dim == d_model."
            )

        # (Removed the default-output_dim and strict input_dim==d_model checks)


    @classmethod
    def from_dict(cls, config_dict: Dict) -> "TransformerConfig":
        """Create a configuration from a dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "TransformerConfig":
        """Load configuration from a JSON file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict:
        """Convert configuration to a dictionary"""
        return {k: v for k, v in self.__dict__.items()}
    
    def save_json(self, file_path: Union[str, Path]) -> None:
        """Save configuration to a JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def __str__(self):
        """String representation of the config"""
        return f"{self.model_name} - {self.d_model}d, {self.n_heads} heads, {self.n_layers} layers"


# Pre-defined configurations
TINY_CONFIG = TransformerConfig(
    vocab_size=5000,
    d_model=64,
    n_heads=2,
    n_layers=2,
    d_ff=128,
    model_name="miniformer-tiny"
)

SMALL_CONFIG = TransformerConfig(
    vocab_size=10000,
    d_model=128,
    n_heads=4,
    n_layers=4,
    d_ff=512,
    model_name="miniformer-small"
)

BASE_CONFIG = TransformerConfig(
    vocab_size=30000,
    d_model=256,
    n_heads=8,
    n_layers=6,
    d_ff=1024,
    model_name="miniformer-base"
)
