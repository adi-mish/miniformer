from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from miniformer.config.model_config import TransformerConfig
from miniformer.model.attention import MultiHeadAttention
from miniformer.model.cache import AttentionList, EncoderPastKeyValues, KeyValueCache
from miniformer.model.feedforward import FeedForward
from miniformer.model.initialization import init_transformer_module


@dataclass(frozen=True)
class EncoderLayerOutput:
    """Explicit output for one encoder layer."""

    hidden_states: torch.Tensor
    self_attention: Optional[torch.Tensor]
    key_values: Optional[KeyValueCache] = None


@dataclass(frozen=True)
class EncoderOutput:
    """Explicit encoder stack output."""

    hidden_states: torch.Tensor
    self_attentions: AttentionList
    past_key_values: Optional[EncoderPastKeyValues] = None

    @property
    def output(self) -> torch.Tensor:
        return self.hidden_states


class EncoderLayer(nn.Module):
    """Transformer encoder layer (pre-norm by default)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.pre_norm = getattr(config, "pre_norm", True)

        self.self_attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_sdpa=getattr(config, "use_sdpa", False),
            rotary_pct=(
                config.rotary_pct if config.position_mode in {"rope", "learned+rope"} else 0.0
            ),
        )
        self.feed_forward = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
        )

        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[KeyValueCache] = None,
        use_cache: bool = False,
    ) -> EncoderLayerOutput:
        """Run one encoder layer."""

        # ── self-attention ────────────────────────────────────────────────
        residual = x
        x_ = self.norm1(x) if self.pre_norm else x
        sa_out, attn, new_key_values = self.self_attention(
            x_, x_, x_, mask, past_key_value, use_cache
        )
        x = residual + self.dropout(sa_out)
        if not self.pre_norm:
            x = self.norm1(x)

        # ── FFN ───────────────────────────────────────────────────────────
        residual = x
        x_ = self.norm2(x) if self.pre_norm else x
        ff_out = self.feed_forward(x_)
        x = residual + self.dropout(ff_out)
        if not self.pre_norm:
            x = self.norm2(x)

        return EncoderLayerOutput(
            hidden_states=x,
            self_attention=attn,
            key_values=new_key_values,
        )


class Encoder(nn.Module):
    """Stack of *n_layers* encoder blocks that supports tokens *or* generic features."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # ― input ​projection / embeddings ―
        self.token_embedding: Optional[nn.Embedding]
        self.input_projection: Optional[nn.Linear]
        if config.input_dim is None:  # NLP path
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.input_projection = None
        else:  # generic feature path
            self.token_embedding = None
            self.input_projection = nn.Linear(config.input_dim, config.d_model)

        self.pos_embedding: Optional[nn.Embedding]
        if config.position_mode in {"learned", "learned+rope"}:
            self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            self.pos_embedding = None
        self.dropout = nn.Dropout(config.dropout)

        # ― transformer blocks ―
        self.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.n_layers))

        # weight init identical to decoder
        self.apply(self._init_weights)

        self.attn_weights: Optional[AttentionList] = None

    # ------------------------------------------------------------------ utils
    def _init_weights(self, module):
        init_transformer_module(module, self.config.initializer_range)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,  # [B, S] int or [B, S, feat]
        mask: Optional[torch.Tensor] = None,  # broadcastable to [B, 1, 1, S]
        *,
        past_key_values: Optional[EncoderPastKeyValues] = None,
        use_cache: bool = False,
    ) -> EncoderOutput:

        B, S = x.size(0), x.size(1)
        device = x.device
        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            first_past = past_key_values[0]
            if first_past is not None:
                past_len = first_past.key.size(2)
        if past_len + S > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {past_len + S} exceeds max_seq_len={self.config.max_seq_len}"
            )

        # input ↦ d_model
        if self.token_embedding is not None:
            x = self.token_embedding(x) * (self.config.d_model**0.5)
        elif self.input_projection is not None:
            if x.dim() != 3 or x.size(-1) != self.config.input_dim:
                raise ValueError("Expected feature tensor of shape [B, S, input_dim]")
            x = self.input_projection(x)
        else:
            raise RuntimeError("Encoder has no input layer (token_embedding or input_projection)")

        # add positions
        if self.pos_embedding is not None:
            positions = (
                torch.arange(past_len, past_len + S, device=device).unsqueeze(0).expand(B, S)
            )
            x = x + self.pos_embedding(positions)
        x = self.dropout(x)

        # run blocks
        attn_weights: AttentionList = []
        if past_key_values is None:
            past_key_values = [None for _ in range(len(self.layers))]
        new_past_key_values: EncoderPastKeyValues = []
        for index, layer in enumerate(self.layers):
            layer_output = layer(
                x,
                mask,
                past_key_value=past_key_values[index],
                use_cache=use_cache,
            )
            x = layer_output.hidden_states
            attn_weights.append(layer_output.self_attention)
            if use_cache:
                new_past_key_values.append(layer_output.key_values)
        self.attn_weights = attn_weights

        return EncoderOutput(
            hidden_states=x,
            self_attentions=attn_weights,
            past_key_values=new_past_key_values if use_cache else None,
        )
