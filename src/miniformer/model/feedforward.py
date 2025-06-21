import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """FFN with GELU / ReLU / SwiGLU (gated)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.activation_name = activation.lower()
        if self.activation_name == "swiglu":  # half projection, double out = 2*d_ff
            self.w12 = nn.Linear(d_model, d_ff * 2)
            self.proj = nn.Linear(d_ff, d_model)
        else:
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.activation_name == "swiglu":
            x1, x2 = self.w12(x).chunk(2, dim=-1)
            return self.proj(self.dropout(F.silu(x1) * x2))
        else:
            act_fn = F.gelu if self.activation_name == "gelu" else F.relu
            return self.linear2(self.dropout(act_fn(self.linear1(x))))