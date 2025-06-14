import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """Token embedding layer"""
    
    def __init__(self, vocab_size, d_model):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        
    def forward(self, x):
        """
        Args:
            x: Input token ids (batch_size, seq_len)
            
        Returns:
            embeddings: Token embeddings (batch_size, seq_len, d_model)
        """
        # Scale embeddings by sqrt(d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        """
        Args:
            d_model: Embedding dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        self.pe: torch.Tensor
        
    def forward(self, x):
        """
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
            
        Returns:
            embeddings: Embeddings with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
