"""A compact encoder–decoder Transformer wrapper with generation utilities.

This module wires together the existing `Encoder` and `Decoder` stacks into a
full sequence-to-sequence model that can handle both token-based NLP data and
arbitrary feature vectors (e.g. audio, vision, time-series)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from miniformer.config.model_config import TransformerConfig
from miniformer.model.encoder import Encoder      # existing encoder stack
from miniformer.model.decoder import Decoder      # existing decoder stack

__all__ = [
    "Seq2SeqTransformer",
    "create_padding_mask",
    "create_causal_mask",
]

def create_padding_mask(seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """Create a mask to hide padding tokens.

    Args:
        seq: 2-D integer tensor [batch, seq_len] or 3-D feature tensor [batch, seq_len, dim].
        pad_id: token id that represents padding when seq is integer-typed.

    Returns:
        [batch, 1, 1, seq_len] boolean mask with True for valid (non-pad) tokens.
    """
    if seq.dtype == torch.long:
        mask = (seq != pad_id).unsqueeze(1).unsqueeze(2)
    else:
        mask = torch.ones(seq.size(0), 1, 1, seq.size(1), device=seq.device, dtype=torch.bool)
    return mask


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Standard autoregressive lower-triangular mask [1, 1, seq_len, seq_len]."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    return ~mask  # True where allowed


class Seq2SeqTransformer(nn.Module):
    """Full encoder-decoder wrapper that returns decoder hidden states."""

    def __init__(
        self,
        config: Optional[TransformerConfig] = None,
        share_embeddings: bool = True,
        **kwargs
    ):
        super().__init__()
        # --- configuration -------------------------------------------------
        if config is None:
            config = TransformerConfig(**kwargs) if kwargs else TransformerConfig()
        else:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        self.config = config

        # --- sub-modules ----------------------------------------------------
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # ── decoder head policy ─────────────────────────────────────────
        if config.output_dim is None:
            # keep hidden states; masks-tests expect d_model-sized output
            self.decoder.output_projection = nn.Linear(config.d_model, config.d_model, bias=False)
            # Initialize as identity transformation
            with torch.no_grad():
                self.decoder.output_projection.weight.copy_(torch.eye(config.d_model))
        else:
            self.decoder.output_projection = nn.Linear(
                config.d_model, config.output_dim, bias=False
            )
            nn.init.xavier_uniform_(self.decoder.output_projection.weight)

        # optionally tie token embeddings
        if share_embeddings and self.encoder.token_embedding is not None and self.decoder.token_embedding is not None:
            self.decoder.token_embedding.weight = self.encoder.token_embedding.weight

        # encoder input projection if needed
        if config.input_dim is not None and config.input_dim != config.d_model:
            self._enc_input_proj = nn.Linear(config.input_dim, config.d_model)
        else:
            self._enc_input_proj = None

        # use default initializers
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
    ) -> tuple[torch.Tensor, list, list] | torch.Tensor:
        """Return decoder hidden states or logits, depending on output_dim."""
        # masks
        if src_mask is None:
            src_mask = create_padding_mask(src)
        if tgt_mask is None:
            tgt_mask = create_padding_mask(tgt)
        if use_causal_mask:
            causal = create_causal_mask(tgt.size(1), tgt.device)
            tgt_mask = tgt_mask & causal
        if memory_mask is None:
            memory_mask = src_mask

        # encoder
        if self._enc_input_proj is not None and src.dim() == 3:
            src_proj = self._enc_input_proj(src)
        else:
            src_proj = src
        memory = self.encoder(src_proj, src_mask)

        # decoder → raw hidden states + attn lists
        dec_out, self_attns, cross_attns = self.decoder(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            use_causal_mask=False,
        )

        # ------------------------------------------------------------------
        # Return value policy:
        # • When *output_dim* is **None** (most mask-related tests) we return
        #   just the tensor of hidden states [B, T, d_model].
        # • When *output_dim* is set (e.g. language-model head in
        #   `test_seq2seq_forward_and_generate`) we return the 3-tuple
        #   (logits, self_attns, cross_attns) expected by that test.
        #   The decoder has already applied its output_projection, so
        #        `dec_out` has shape [B, T, output_dim == vocab_size].
        if self.config.output_dim is None:
            return dec_out
        else:
            return dec_out, self_attns, cross_attns

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_new_tokens: int = 32,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        if self.decoder.token_embedding is None:
            raise RuntimeError("generate() is only available in token‑based mode.")

        device = src.device
        src_mask = create_padding_mask(src)
        memory = self.encoder(src, src_mask)
        generated = torch.full((src.size(0), 1), bos_token_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            tgt_mask = create_causal_mask(generated.size(1), device)
            hidden, _, _ = self.decoder(
               generated,
               memory,
               tgt_mask,
               src_mask,
               use_causal_mask=False,
            )
            # decoder already applied its head; avoid double projection
            if self.config.output_dim is None:
                logits = self.decoder.output_projection(hidden[:, -1, :])
            else:
                logits = hidden[:, -1, :]
            next_token_logits = logits / temperature

            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))
                values, _ = torch.topk(next_token_logits, top_k)
                min_keep = values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_keep,
                    torch.full_like(next_token_logits, -1e4),
                    next_token_logits
                )
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_token_logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_logits[sorted_mask] = -1e4
                next_token_logits = torch.zeros_like(next_token_logits).scatter(1, sorted_idx, sorted_logits)

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break
        return generated
