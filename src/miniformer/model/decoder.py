import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from miniformer.model.attention import MultiHeadAttention
from miniformer.model.feedforward import FeedForward
from miniformer.config.model_config import TransformerConfig


class DecoderLayer(nn.Module):
    """Transformer decoder layer with self-attention and cross-attention"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.pre_norm = getattr(config, "pre_norm", True)
        
        self.self_attention = MultiHeadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        self.cross_attention = MultiHeadAttention(
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
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        past_self: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_cross: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        # choose norm order
        residual = x
        if self.pre_norm:
            x_ = self.norm1(x)
        else:
            x_ = x

        attn_out, self_attn, new_self = self.self_attention(
            x_, x_, x_, self_attn_mask, past_self, use_cache
        )
        x = residual + self.dropout(attn_out)
        if not self.pre_norm:
            x = self.norm1(x)

        # --- cross (no caching) ------------------------------------------
        residual = x
        if self.pre_norm:
            x_ = self.norm2(x)
        else:
            x_ = x
        # we do NOT cache encoder k/v, so pass past_kv=None and use_cache=False
        cross_out, cross_attn, _ = self.cross_attention(
            q=x_,
            k=encoder_output,
            v=encoder_output,
            mask=cross_attn_mask,
            past_kv=None,
            use_cache=False,
        )
        x = residual + self.dropout(cross_out)
        if not self.pre_norm:
            x = self.norm2(x)
        # new_cross is always None now
        new_cross = None

        # --- ffn -----------------------------------------------------------
        residual = x
        if self.pre_norm:
            x_ = self.norm3(x)
        else:
            x_ = x
        ff_out = self.feed_forward(x_)
        x = residual + self.dropout(ff_out)
        if not self.pre_norm:
            x = self.norm3(x)

        return x, self_attn, cross_attn, new_self, new_cross

class Decoder(nn.Module):
    """Transformer decoder stack supporting various data types"""
    # allow output_projection to be any module (Linear, Identity, etc.)
    output_projection: nn.Module
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Input projection for feature vectors or token embeddings
        if config.input_dim is not None:
            # For direct feature input (time series, sensor data, etc.)
            self.input_projection = nn.Linear(config.input_dim, config.d_model)
            self.token_embedding = None
        else:
            # For token-based input (NLP tasks)
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
            self.input_projection = None
            
        # Positional encoding - learnable for flexibility with different sequence types
        self.position_encoding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(config)
            for _ in range(config.n_layers)
        ])
        
        # Output projection based on task type
        if self.token_embedding is not None:
            # Language modeling task
            self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        else:
            # Classification or regression task
            if config.output_dim is None:
                raise ValueError("output_dim must be specified for non-token-based tasks.")
            self.output_projection = nn.Linear(config.d_model, config.output_dim)
        
        # Apply weight initialization
        self.apply(self._init_weights)

        self.self_attentions: Optional[List[torch.Tensor]] = None
        self.cross_attentions: Optional[List[torch.Tensor]] = None
        
    def _init_weights(self, module):
        """Initialize weights following transformer conventions"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal (lower triangular) attention mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        *,
        past_key_values: Optional[
            List[Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]],
                       Optional[Tuple[torch.Tensor, torch.Tensor]]]]
        ] = None,
        use_cache: bool = False,
        return_hidden: bool = False,              # ← NEW
    ):
        """
        Returns
        -------
        (output, self_attns, cross_attns [, new_past_kv])
        """
        batch_size, seq_len = x.size(0), x.size(1)
        device = x.device

        # ── token/feature input → d_model ───────────────────────────────
        if self.token_embedding is not None:
            x = self.token_embedding(x) * (self.config.d_model ** 0.5)
        elif self.input_projection is not None:
            if x.dim() != 3 or x.size(-1) != self.config.input_dim:
                raise ValueError(f"Expected feature tensor of shape [B, S, {self.config.input_dim}]")
            x = self.input_projection(x)
        else:
            raise RuntimeError("Decoder has no input layer")

        # positional encodings ------------------------------------------------
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.dropout(x + self.position_encoding(positions))

        if use_causal_mask and self_attn_mask is None:
            self_attn_mask = self.create_causal_mask(seq_len, device)

        self_attentions, cross_attentions = [], []
        if past_key_values is None:
            past_key_values = [(None, None) for _ in range(len(self.layers))]
        new_past_kv: List[Tuple[Tuple, Tuple]] = []

        # ── transformer layers ───────────────────────────────────────────
        for i, layer in enumerate(self.layers):
            past_self, past_cross = past_key_values[i]
            x, self_attn, cross_attn, new_self, new_cross = layer(
                x, encoder_output,
                self_attn_mask, cross_attn_mask,
                past_self, past_cross, use_cache,
            )
            self_attentions.append(self_attn)
            cross_attentions.append(cross_attn)
            if use_cache:
                new_past_kv.append((new_self, new_cross))

        # ── final projection (optional) ──────────────────────────────────
        output = x if return_hidden else self.output_projection(x)

        return (output, self_attentions, cross_attentions, new_past_kv) if use_cache \
               else (output, self_attentions, cross_attentions)

    def get_attention_weights(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get attention weights from the last forward pass"""
        # This would need to be called after a forward pass
        if self.self_attentions is None or self.cross_attentions is None:
            raise RuntimeError("Attention weights not available. Run a forward pass first.")
        return self.self_attentions, self.cross_attentions
