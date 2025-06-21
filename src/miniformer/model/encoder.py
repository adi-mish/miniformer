from __future__ import annotations
from typing import Optional, List, Tuple

import torch
import torch.nn as nn

from miniformer.config.model_config import TransformerConfig
from miniformer.model.attention import MultiHeadAttention
from miniformer.model.feedforward import FeedForward


class EncoderLayer(nn.Module):
    """Transformer encoder layer (pre-norm by default)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.pre_norm = getattr(config, "pre_norm", True)

        self.self_attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_sdpa=getattr(config, "use_sdpa", True),
            rotary_pct=getattr(config, "rotary_pct", 0.0),
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (output, attn_weights)."""

        # ── self-attention ────────────────────────────────────────────────
        residual = x
        x_ = self.norm1(x) if self.pre_norm else x
        sa_out, attn, _ = self.self_attention(x_, x_, x_, mask)
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

        return x, attn


class Encoder(nn.Module):
    """Stack of *n_layers* encoder blocks that supports tokens *or* generic features."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # ― input ​projection / embeddings ―
        if config.input_dim is None:                           # NLP path
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.input_projection = None
        else:                                                  # generic feature path
            self.token_embedding = None
            self.input_projection = nn.Linear(config.input_dim, config.d_model)

        # Learned positional embeddings for maximum flexibility
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # ― transformer blocks ―
        self.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.n_layers))

        # weight init identical to decoder
        self.apply(self._init_weights)

        self.attn_weights: Optional[List[torch.Tensor]] = None

    # ------------------------------------------------------------------ utils
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,                       # [B, S] int or [B, S, feat]
        mask: Optional[torch.Tensor] = None,   # broadcastable to [B, 1, 1, S]
    ) -> torch.Tensor:

        B, S = x.size(0), x.size(1)
        device = x.device

        # input ↦ d_model
        if self.token_embedding is not None:
            x = self.token_embedding(x) * (self.config.d_model ** 0.5)
        elif self.input_projection is not None:
            if x.dim() != 3 or x.size(-1) != self.config.input_dim:
                raise ValueError("Expected feature tensor of shape [B, S, input_dim]")
            x = self.input_projection(x)
        else:
            raise RuntimeError("Encoder has no input layer (token_embedding or input_projection)")

        # add positions
        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
        x = x + self.pos_embedding(positions)
        x = self.dropout(x)

        # run blocks
        self.attn_weights = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            self.attn_weights.append(attn)

        return x