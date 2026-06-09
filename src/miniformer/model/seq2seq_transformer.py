"""A compact encoder–decoder Transformer wrapper with generation utilities.

This module wires together the existing `Encoder` and `Decoder` stacks into a
full sequence-to-sequence model that can handle both token-based NLP data and
arbitrary feature vectors (e.g. audio, vision, time-series)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from miniformer.config.model_config import TransformerConfig
from miniformer.model.cache import DecoderPastKeyValues
from miniformer.model.decoder import Decoder
from miniformer.model.encoder import Encoder  # existing encoder stack
from miniformer.model.masks import (
    causal_mask,
    combine_masks,
    padding_mask,
)
from miniformer.model.outputs import Seq2SeqModelOutput

if TYPE_CHECKING:
    from miniformer.inspect import TransformerTrace

__all__ = [
    "Seq2SeqModelOutput",
    "Seq2SeqTransformer",
]


class Seq2SeqTransformer(nn.Module):
    """Full encoder-decoder wrapper with explicit model outputs."""

    def __init__(
        self, config: Optional[TransformerConfig] = None, share_embeddings: bool = True, **kwargs
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

        # optionally tie token embeddings
        if (
            share_embeddings
            and self.encoder.token_embedding is not None
            and self.decoder.token_embedding is not None
        ):
            self.decoder.token_embedding.weight = self.encoder.token_embedding.weight

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
    ) -> Seq2SeqModelOutput:
        """Run encoder-decoder forward pass and return the configured output mode."""
        # ── build masks ─────────────────────────────────────────────────
        if src_mask is None:
            src_mask = padding_mask(src)
        if tgt_mask is None:
            tgt_mask = padding_mask(tgt)
        if use_causal_mask:
            tgt_mask = combine_masks(tgt_mask, causal_mask(tgt.size(1), device=tgt.device))
            assert tgt_mask is not None
        if memory_mask is None:
            memory_mask = src_mask

        # ── encode ─────────────────────────────────────────────────────
        memory = self.encoder(src, src_mask).hidden_states

        # ── decode ─────────────────────────────────────────────────────
        decoder_output = self.decoder(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            use_causal_mask=False,
        )

        dec_out = decoder_output.output
        return Seq2SeqModelOutput(
            logits=dec_out if self.config.output_mode == "vocab" else None,
            hidden_states=dec_out if self.config.output_mode == "hidden" else None,
            projection=dec_out if self.config.output_mode == "projection" else None,
            self_attentions=decoder_output.self_attentions,
            cross_attentions=decoder_output.cross_attentions,
        )

    def trace(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        *,
        top_k: int = 5,
        compare_cache: bool = False,
    ) -> TransformerTrace:
        """Capture a structured inspection trace for one seq2seq forward pass."""
        from miniformer.inspect import capture_transformer_trace

        return capture_transformer_trace(
            self,
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            use_causal_mask=use_causal_mask,
            top_k=top_k,
            compare_cache=compare_cache,
        )

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
        if self.config.output_mode != "vocab":
            raise RuntimeError("generate() requires output_mode='vocab'")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        device = src.device
        src_mask = padding_mask(src)
        memory = self.encoder(src, src_mask).hidden_states
        generated = torch.full((src.size(0), 1), bos_token_id, dtype=torch.long, device=device)

        # Implement caching for faster generation
        past_key_values: Optional[DecoderPastKeyValues] = None
        use_cache = True

        for _ in range(max_new_tokens):
            # Use caching for more efficient generation
            if use_cache:
                decoder_output = self.decoder(
                    generated if past_key_values is None else generated[:, -1:],
                    memory,
                    None,
                    src_mask,
                    use_causal_mask=True,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                dec_out = decoder_output.output
                past_key_values = decoder_output.past_key_values
            else:
                decoder_output = self.decoder(
                    generated,
                    memory,
                    None,
                    src_mask,
                    use_causal_mask=True,
                )
                dec_out = decoder_output.output

            logits = dec_out[:, -1, :]
            logits = logits / temperature

            # Apply top-k and top-p filtering
            if top_k > 0:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k)
                min_keep = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_keep, torch.full_like(logits, -1e4), logits)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                remove_mask = torch.zeros_like(logits, dtype=torch.bool)
                remove_mask.scatter_(dim=-1, index=sorted_idx, src=sorted_mask)
                logits = logits.masked_fill(remove_mask, -1e4)

            # Sample from the filtered distribution
            probs = logits.softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the predicted token to the sequence
            generated = torch.cat((generated, next_token), dim=1)
            if (next_token == eos_token_id).all():
                break

        return generated[:, 1:]  # Exclude the initial BOS token
