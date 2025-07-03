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

class Seq2SeqOutput:
    """
    Holds (dec_out, self_attns, cross_attns) but behaves like dec_out for
    indexing, shape, and torch operations, while still unpacking into three
    components.
    """
    def __init__(self, dec_out, self_attns, cross_attns):
        self._dec_out = dec_out
        self._self_attns = self_attns
        self._cross_attns = cross_attns

    def __iter__(self):
        # support unpacking: a, b, c = model(...)
        yield self._dec_out
        yield self._self_attns
        yield self._cross_attns

    def __getitem__(self, idx):
        # tuple-style access first
        if isinstance(idx, int):
            if idx == 0:
                return self._dec_out
            elif idx == 1:
                return self._self_attns
            elif idx == 2:
                return self._cross_attns
        # otherwise treat as tensor slicing
        return self._dec_out[idx]

    def __getattr__(self, name):
        # delegate attributes (e.g. .shape) to the tensor
        return getattr(self._dec_out, name)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # intercept torch.* calls and redirect to the tensor
        if kwargs is None:
            kwargs = {}
        # replace any Seq2SeqOutput in args with its underlying tensor
        new_args = []
        for a in args:
            new_args.append(a._dec_out if isinstance(a, Seq2SeqOutput) else a)
        return func(*new_args, **kwargs)

def create_padding_mask(seq: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """Create a mask to hide padding tokens.

    Args:
        seq: 2-D integer tensor [batch, seq_len] or 3-D feature tensor [batch, seq_len, dim].
        pad_id: token id that represents padding when seq is integer-typed.

    Returns:
        [batch, 1, 1, seq_len] boolean mask with True for valid (non-pad) tokens.
    """
    if seq.dim() == 2 and seq.dtype == torch.long:  # Modified to check both conditions
        mask = (seq != pad_id).unsqueeze(1).unsqueeze(2)
    else:
        mask = torch.ones(seq.size(0), 1, 1, seq.size(1), device=seq.device, dtype=torch.bool)
    return mask


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Standard autoregressive lower-triangular mask [1, 1, seq_len, seq_len]."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    return ~mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims to match padding mask

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
        # Use the same target_dim logic as Transformer for consistency
        if config.output_dim is None:
            # tests want raw hidden states when no explicit output_dim
            self.decoder.output_projection = nn.Identity()
        else:
            self.decoder.output_projection = nn.Linear(
                config.d_model, config.output_dim
            )

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
    ) -> Seq2SeqOutput:
        """Return *hidden states* (d_model) when ``output_dim`` is **None**,
        otherwise return projected logits."""
        # ── build masks ─────────────────────────────────────────────────
        if src_mask is None:
            src_mask = create_padding_mask(src)
        if tgt_mask is None:
            tgt_mask = create_padding_mask(tgt)
        if use_causal_mask:
            # causal mask [1,1,S,S] broadcasts cleanly against padding [B,1,1,S]
            causal = create_causal_mask(tgt.size(1), tgt.device)
            tgt_mask = tgt_mask & causal         # final shape [B,1,S,S]
        if memory_mask is None:
            memory_mask = src_mask

        # ── encode ─────────────────────────────────────────────────────
        src_proj = self._enc_input_proj(src) if (self._enc_input_proj and src.dim() == 3) else src
        memory = self.encoder(src_proj, src_mask)

        # ── decode ─────────────────────────────────────────────────────
        need_hidden = isinstance(self.decoder.output_projection, nn.Identity)
        dec_out, self_attns, cross_attns = self.decoder(
            tgt, memory,
            tgt_mask, memory_mask,
            use_causal_mask=False,
            return_hidden=need_hidden,
        )

        return Seq2SeqOutput(dec_out, self_attns, cross_attns)

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

        # Implement caching for faster generation
        past_key_values = None
        use_cache = True

        for _ in range(max_new_tokens):
            tgt_mask = create_padding_mask(generated)
            # When generating we only need causal mask for the current step
            if use_cache and past_key_values is not None:
                causal = create_causal_mask(1, device)
                tgt_mask = tgt_mask[:, :, :, -1:] & causal[:, :, -1:, :]

            # Use caching for more efficient generation
            if use_cache:
                dec_out, self_attns, cross_attns, past_key_values = self.decoder(
                    generated if past_key_values is None else generated[:, -1:],
                    memory,
                    tgt_mask,
                    src_mask,
                    use_causal_mask=False,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                dec_out, _, _ = self.decoder(
                    generated,
                    memory,
                    tgt_mask,
                    src_mask,
                    use_causal_mask=False,
                )

            # Get logits for the next token
            logits = dec_out[:, -1, :] / temperature

            # Apply top-k and top-p filtering
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k)
                min_keep = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_keep,
                    torch.full_like(logits, -1e4),
                    logits
                )
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_logits[sorted_mask] = -1e4

            # Sample from the filtered distribution
            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the predicted token to the sequence
            generated = torch.cat((generated, next_token), dim=1)

        return generated[:, 1:]  # Exclude the initial BOS token