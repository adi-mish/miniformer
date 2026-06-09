"""
Multi‑head attention with optional
• PyTorch‑2 scaled‑dot‑product attention (SDPA / FlashAttention backend)
• Rotary positional embeddings (RoPE)
• key‑value cache for fast autoregressive decoding

Backwards‑compatible with the original forward signature.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Utility used by RoPE – slice and rotate last dim (= head‑dim)."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(t: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to q or k."""
    # t: [B, heads, L, D]
    t_ = (t * cos) + (_rotate_half(t) * sin)
    return t_


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_sdpa: bool = False,  # SDPA requires PyTorch 2.0+
        rotary_pct: float = 0.0,  # 0 = disabled, 1 = apply to full head‑dim
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        if rotary_pct < 0 or rotary_pct > 1:
            raise ValueError("rotary_pct must be in the range [0, 1]")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_sdpa = use_sdpa and hasattr(F, "scaled_dot_product_attention")
        self.rotary_dim = int(self.d_k * rotary_pct)
        self.rotary_dim -= self.rotary_dim % 2

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # pre‑compute sin / cos cache for RoPE – small for “mini” models
        if self.rotary_dim > 0:
            inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self._rope_cache: dict[
                tuple[int, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]
            ] = {}
        else:
            self.inv_freq = None
            self._rope_cache = {}

    # ------------------------------------------------------------------ utils
    def _rope(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Compute / fetch cached sinusoid and apply RoPE on first *rotary_dim*."""
        seq_len = x.shape[2]
        total_len = offset + seq_len
        cache_key = (total_len, x.device, x.dtype)
        if cache_key not in self._rope_cache:
            t = torch.arange(total_len, device=x.device, dtype=torch.float32)
            freqs = torch.einsum("i , j -> i j", t, self.inv_freq)  # [L, D/2]
            emb = torch.cat([freqs, freqs], dim=-1)  # [L, D]
            sin = emb.sin()[None, None, :, :].to(dtype=x.dtype)
            cos = emb.cos()[None, None, :, :].to(dtype=x.dtype)
            self._rope_cache[cache_key] = (sin, cos)
        sin, cos = self._rope_cache[cache_key]
        sin = sin[:, :, offset:total_len, :]
        cos = cos[:, :, offset:total_len, :]
        # only apply to first rotary_dim slice
        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]
        x_rot = apply_rope(x_rot, sin[..., : self.rotary_dim], cos[..., : self.rotary_dim])
        return torch.cat([x_rot, x_pass], dim=-1)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        q: torch.Tensor,  # [B, L, D]
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,  # broadcastable to [B, heads, L, S]
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Return (output, attn, new_kv) where *new_kv* is None unless use_cache=True."""
        B, L, _ = q.shape

        # project & reshape --------------------------------------------------
        q = self.wq(q).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # [B, H, L, D]
        k = self.wk(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # RoPE --------------------------------------------------------------
        past_len = past_kv[0].shape[2] if past_kv is not None else 0
        if self.rotary_dim > 0:
            q = self._rope(q, offset=past_len)
            k = self._rope(k, offset=past_len)

        # kv‑cache concat ----------------------------------------------------
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)  # time dim = 2 after transpose
            v = torch.cat([pv, v], dim=2)

        # SDPA or manual attention -----------------------------------------
        if self.use_sdpa:
            dropout_p = self.dropout.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
            attn = None  # PyTorch does not return weights
            out = out.transpose(1, 2)  # [B, L, H, D]
        else:
            scale = 1.0 / math.sqrt(self.d_k)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, S]
            if mask is not None:
                if mask.dtype is not torch.bool:
                    raise TypeError("attention mask must be a boolean tensor")
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            attn = F.softmax(scores, dim=-1)
            if mask is not None:
                attn = attn.masked_fill(~mask, 0.0)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v).transpose(1, 2)  # [B, L, H, D]

        # merge heads & final proj ------------------------------------------
        out = out.contiguous().view(B, L, self.d_model)
        out = self.wo(out)

        new_kv = (k, v) if use_cache else None
        return out, attn, new_kv
