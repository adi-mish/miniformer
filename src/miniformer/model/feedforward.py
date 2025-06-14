import torch.nn as nn


class FeedForward(nn.Module):
    """Feed-forward network used in transformer"""
    
    def __init__(self, d_model, d_ff, dropout=0.1, activation="gelu"):
        """
        Args:
            d_model: Input and output dimension
            d_ff: Hidden dimension
            dropout: Dropout probability
            activation: Activation function ("gelu" or "relu")
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Select activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            output: Transformed tensor (batch_size, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
