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

        # --- cross ---------------------------------------------------------
        residual = x
        if self.pre_norm:
            x_ = self.norm2(x)
        else:
            x_ = x
        cross_out, cross_attn, new_cross = self.cross_attention(
            q=x_,
            k=encoder_output,
            v=encoder_output,
            mask=cross_attn_mask,
            past_kv=past_cross,
            use_cache=use_cache,
        )
        x = residual + self.dropout(cross_out)
        if not self.pre_norm:
            x = self.norm2(x)

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
        use_causal_mask: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through decoder
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim] or [batch_size, seq_len] for tokens
            encoder_output: Output from encoder [batch_size, encoder_seq_len, d_model]
            self_attn_mask: Optional mask for self-attention
            cross_attn_mask: Optional mask for cross-attention
            use_causal_mask: Whether to apply causal masking for autoregressive tasks
            
        Returns:
            output: Final output tensor
            self_attentions: List of self-attention weights from each layer
            cross_attentions: List of cross-attention weights from each layer
        """
        batch_size = x.size(0)
        seq_len = x.size(1)
        device = x.device
        
        # Convert input to embeddings
        if self.token_embedding is not None and x.dim() == 2:
            # Token-based input (NLP)
            x = self.token_embedding(x)
        elif self.input_projection is not None:
            # Feature-based input (time series, sensor data, etc.)
            x = self.input_projection(x)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        x = x + self.position_encoding(positions)
        x = self.dropout(x)
        
        # Create causal mask for autoregressive tasks if needed
        if use_causal_mask and self_attn_mask is None:
            self_attn_mask = self.create_causal_mask(seq_len, device)

        # Store attention weights for analysis/visualization
        self_attentions: List[torch.Tensor] = []
        cross_attentions: List[torch.Tensor] = []

        # Apply decoder layers
        for layer in self.layers:
            x, self_attn, cross_attn = layer(
                x, encoder_output, self_attn_mask, cross_attn_mask
            )
            self_attentions.append(self_attn)
            cross_attentions.append(cross_attn)
        self.self_attentions = self_attentions
        self.cross_attentions = cross_attentions

        # Final output projection
        output = self.output_projection(x)
        return output, self_attentions, cross_attentions
    
    def get_attention_weights(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Get attention weights from the last forward pass"""
        # This would need to be called after a forward pass
        if self.self_attentions is None or self.cross_attentions is None:
            raise RuntimeError("Attention weights not available. Run a forward pass first.")
        return self.self_attentions, self.cross_attentions
