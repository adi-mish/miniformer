import torch
import torch.nn as nn

from miniformer.model.attention import MultiHeadAttention
from miniformer.model.embedding import TokenEmbedding, PositionalEncoding
from miniformer.model.feedforward import FeedForward


class EncoderLayer(nn.Module):
    """Transformer encoder layer"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward network hidden dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
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
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout=0.1, max_seq_len=5000):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward network hidden dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.attention_weights = None
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input token ids (batch_size, seq_len)
            mask: Attention mask (batch_size, 1, 1, seq_len)
            
        Returns:
            output: Encoded representation (batch_size, seq_len, d_model)
        """
        # Get token embeddings and add positional encoding
        x = self.token_embedding(x)
        x = self.position_encoding(x)
        
        # Store attention weights for visualization
        self.attention_weights = []
        
        # Apply encoder layers
        for layer in self.layers:
            x, attention = layer(x, mask)
            self.attention_weights.append(attention)
            
        return x


class Transformer(nn.Module):
    """Simple transformer model for sequence classification or prediction tasks"""
    
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=3, d_ff=256, dropout=0.1, max_seq_len=5000):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward network hidden dimension
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_seq_len)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input token ids (batch_size, seq_len)
            mask: Attention mask (batch_size, 1, 1, seq_len)
            
        Returns:
            output: Output logits (batch_size, seq_len, vocab_size)
        """
        # Create mask if not provided
        if mask is None and x is not None:
            mask = self._create_mask(x)
            
        # Encode input
        encoded = self.encoder(x, mask)
        
        # Project to vocabulary
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
