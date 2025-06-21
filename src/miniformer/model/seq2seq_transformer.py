"""A compact encoder–decoder Transformer wrapper with generation utilities.

This module wires together the existing ``Encoder`` and ``Decoder`` stacks into a
full sequence‑to‑sequence model that can handle both token‑based NLP data and
arbitrary feature vectors (e.g. audio, vision, time‑series).  It also ships
helpers for common mask creation and a greedy ``generate`` method.  Beam search
and kv‑cache can be added later without changing external APIs.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from miniformer.config.model_config import TransformerConfig
from miniformer.model.encoder import Encoder      # existing decoder stack
from miniformer.model.decoder import Decoder      # existing decoder stack

__all__ = [
    "Seq2SeqTransformer",
    "create_padding_mask",
    "create_causal_mask",
]

def create_padding_mask(seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """Create a mask to hide *padding* tokens.

    Args:
        seq: 2‑D integer tensor ``[batch, seq_len]`` or **3‑D** feature tensor
              ``[batch, seq_len, dim]`` (non‑integer).  For the feature case all
              positions are considered non‑padding.
        pad_id: token id that represents padding when *seq* is integer‑typed.
    Returns:
        ``[batch, 1, 1, seq_len]`` boolean mask broadcasting‑friendly with
        ``True`` for *valid* (non‑pad) tokens.
    """
    if seq.dtype == torch.long:
        mask = (seq != pad_id).unsqueeze(1).unsqueeze(2)
    else:
        mask = torch.ones(seq.size(0), 1, 1, seq.size(1), device=seq.device, dtype=torch.bool)
    return mask

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Standard autoregressive lower‑triangular mask ``[1, 1, seq_len, seq_len]``."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    return ~mask  # True where allowed


class Seq2SeqTransformer(nn.Module):
    """Full encoder‑decoder wrapper.

    Attributes
    ----------
    config: ``TransformerConfig``
    encoder: ``Encoder`` stack
    decoder: ``Decoder`` stack
    """

    def __init__(self, config: Optional[TransformerConfig] = None, share_embeddings: bool = True, **kwargs):
        super().__init__()

        # --- configuration -------------------------------------------------
        if config is None:
            config = TransformerConfig(**kwargs) if kwargs else TransformerConfig()
        else:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        self.config = config

        # --- sub‑modules ----------------------------------------------------
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Optionally tie token embeddings (weight sharing) for NLP tasks
        if share_embeddings and self.encoder.token_embedding is not None and self.decoder.token_embedding is not None:
            self.decoder.token_embedding.weight = self.encoder.token_embedding.embedding.weight  # type: ignore[attr-defined]

        # In feature‑vector mode (non‑NLP), allow *input_dim != d_model* by adding a projection layer.
        if config.input_dim is not None and config.input_dim != config.d_model:
            self._enc_input_proj = nn.Linear(config.input_dim, config.d_model)
        else:
            self._enc_input_proj = None

        # Final task head: reuse decoder.output_projection for LM; otherwise a dedicated linear layer.
        if self.decoder.token_embedding is None:  # feature or generic task
            if config.output_dim is None:
                raise ValueError("output_dim must be specified for generic tasks when token embeddings are disabled.")
            self.task_head = nn.Linear(config.d_model, config.output_dim)
        else:  # LM – decoder already has projection tied or separate
            self.task_head = self.decoder.output_projection

        # share initialiser util
        self.apply(self._init_weights)

    # ---------------------------------------------------------------------
    # weight init helpers
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    # ---------------------------------------------------------------------
    # forward pass
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Standard forward function for *training* or teacher‑forced decoding.

        Parameters
        ----------
        src : tensor
            Source sequence.  Either token ids ``[B, S]`` (``dtype=long``) or feature
            tensor ``[B, S, input_dim]``.
        tgt : tensor
            Target sequence (shifted‑right when training LM).  Same shape rules as *src*.
        src_mask / tgt_mask / memory_mask : optional boolean masks broadcastable to
            attention shapes.
        use_causal_mask : whether to create causal mask for *tgt* automatically if
            *tgt_mask* is absent.
        Returns
        -------
        logits : ``[B, T, output_dim]``
        self_attns, cross_attns : lists of attention matrices from decoder layers.
        """
        # --- masks ---------------------------------------------------------
        if src_mask is None:
            src_mask = create_padding_mask(src)
        if tgt_mask is None:
            # padding mask first
            tgt_mask = create_padding_mask(tgt)
        if use_causal_mask:
            causal = create_causal_mask(tgt.size(1), tgt.device)
            tgt_mask = tgt_mask & causal  # combine padding + causal
        if memory_mask is None:
            memory_mask = src_mask  # reuse src padding mask

        # --- encoder -------------------------------------------------------
        if self._enc_input_proj is not None and src.dim() == 3:
            src_proj = self._enc_input_proj(src)
        else:
            src_proj = src
        memory = self.encoder(src_proj, src_mask)

        # --- decoder -------------------------------------------------------
        dec_out, self_atts, cross_atts = self.decoder(
            tgt, memory, tgt_mask, memory_mask, use_causal_mask=False  # we already merged causal
        )

        # --- task head -----------------------------------------------------
        logits = self.task_head(dec_out)
        return logits, self_atts, cross_atts

    # ---------------------------------------------------------------------
    # generation interface (greedy)
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
        memory_mask = src_mask

        generated = torch.full((src.size(0), 1), bos_token_id, dtype=torch.long, device=device)
        past_kv: Optional[List[Tuple[Tuple, Tuple]]] = None

        for _ in range(max_new_tokens):
            tgt_mask = create_causal_mask(generated.size(1), device)
            logits, _, _, past_kv = self.decoder(
                generated,
                memory,
                tgt_mask,
                memory_mask,
                past_key_values=past_kv,
                use_causal_mask=False,
                use_cache=True,
            )
            next_token_logits = logits[:, -1, :] / temperature

            # top‑k / nucleus (top‑p) sampling
            if top_k > 0:
                top_k = min(top_k, next_token_logits.size(-1))  # guard
                values, _ = torch.topk(next_token_logits, top_k)
                min_keep = values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_keep, torch.full_like(next_token_logits, -1e4), next_token_logits
                )
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_token_logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_logits[sorted_mask] = -1e4
                next_token_logits = torch.zeros_like(next_token_logits).scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)  # greedy inside filtered set
            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return generated
