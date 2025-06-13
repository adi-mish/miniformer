import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism as described in 'Attention is All You Need' paper"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model: Embedding dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V and output
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query (batch_size, seq_len, d_model)
            k: Key (batch_size, seq_len, d_model)
            v: Value (batch_size, seq_len, d_model)
            mask: Mask for padding (batch_size, 1, 1, seq_len)
            
        Returns:
            output: Self-attention output
            attention: Attention weights
        """
        batch_size = q.size(0)
        
        # Linear projections and reshape to (batch_size, n_heads, seq_len, d_k)
        q = self.wq(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Compute weighted sum
        output = torch.matmul(attention, v)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.wo(output)
        
        return output, attention
