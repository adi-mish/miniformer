import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Union


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
    pre_norm: bool = True
    use_sdpa: bool = False
    causal: bool = True  # Token Transformer defaults to autoregressive self-attention
    rotary_pct: float = 0.0  # 0 = disabled, 1 = full head dimension
    output_mode: Literal["hidden", "vocab", "projection"] = "hidden"
    position_mode: Literal["learned", "rope", "learned+rope"] = "learned"

    # Input/Output dimensions
    input_dim: Optional[int] = None  # If provided, model accepts feature vectors directly
    output_dim: Optional[int] = (
        None  # If provided, model projects to this dimension instead of vocab_size
    )

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
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.rotary_pct < 0 or self.rotary_pct > 1:
            raise ValueError("rotary_pct must be in the range [0, 1]")

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

        self.activation = self.activation.lower()
        valid_activations = {"gelu", "relu", "swiglu"}
        if self.activation not in valid_activations:
            raise ValueError(f"Unknown activation: {self.activation}")

        valid_output_modes = {"hidden", "vocab", "projection"}
        if self.output_mode not in valid_output_modes:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")
        if self.output_mode == "hidden" and self.output_dim is not None:
            raise ValueError("output_mode='hidden' requires output_dim=None")
        if self.output_mode == "vocab":
            if self.input_dim is not None:
                raise ValueError("output_mode='vocab' is only valid for token inputs")
            if self.output_dim is not None:
                raise ValueError("output_mode='vocab' requires output_dim=None")
        if self.output_mode == "projection" and self.output_dim is None:
            raise ValueError("output_mode='projection' requires output_dim")

        valid_position_modes = {"learned", "rope", "learned+rope"}
        if self.position_mode not in valid_position_modes:
            raise ValueError(f"Unknown position_mode: {self.position_mode}")

        if self.position_mode == "learned" and self.rotary_pct != 0:
            raise ValueError("position_mode='learned' requires rotary_pct=0")
        if self.position_mode in {"rope", "learned+rope"} and self.rotary_pct <= 0:
            raise ValueError(f"position_mode='{self.position_mode}' requires rotary_pct > 0")

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "TransformerConfig":
        """Create a configuration from a dictionary"""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "TransformerConfig":
        """Load configuration from a JSON file"""
        with open(file_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict:
        """Convert configuration to a dictionary"""
        return {k: v for k, v in self.__dict__.items()}

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Save configuration to a JSON file"""
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def __str__(self):
        """String representation of the config"""
        return f"{self.model_name} - {self.d_model}d, {self.n_heads} heads, {self.n_layers} layers"


# Pre-defined configurations
TINY_CONFIG = TransformerConfig(
    vocab_size=5000, d_model=64, n_heads=2, n_layers=2, d_ff=128, model_name="miniformer-tiny"
)

SMALL_CONFIG = TransformerConfig(
    vocab_size=10000, d_model=128, n_heads=4, n_layers=4, d_ff=512, model_name="miniformer-small"
)

BASE_CONFIG = TransformerConfig(
    vocab_size=30000, d_model=256, n_heads=8, n_layers=6, d_ff=1024, model_name="miniformer-base"
)
