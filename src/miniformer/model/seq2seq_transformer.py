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
from miniformer.model.generation import GenerationConfig, sample_next_token
from miniformer.model.masks import (
    causal_mask,
    combine_masks,
    padding_mask,
)
from miniformer.model.outputs import Seq2SeqModelOutput

if TYPE_CHECKING:
    from miniformer.inspect import TransformerTrace

__all__ = [
    "GenerationConfig",
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
        include_raw_attention: bool = False,
        include_logits: bool = True,
        max_report_tokens: int = 64,
        max_report_heads: int = 8,
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
            include_raw_attention=include_raw_attention,
            include_logits=include_logits,
            max_report_tokens=max_report_tokens,
            max_report_heads=max_report_heads,
            compare_cache=compare_cache,
        )

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_new_tokens: int = 32,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        use_cache: bool = True,
        generation_config: Optional[GenerationConfig] = None,
    ) -> torch.Tensor:
        if self.decoder.token_embedding is None:
            raise RuntimeError("generate() is only available in token‑based mode.")
        if self.config.output_mode != "vocab":
            raise RuntimeError("generate() requires output_mode='vocab'")
        if generation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_cache=use_cache,
            )
        generation_config.validate(self.config.vocab_size)

        device = src.device
        src_mask = padding_mask(src)
        memory = self.encoder(src, src_mask).hidden_states
        generated = torch.full(
            (src.size(0), 1),
            generation_config.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        if generation_config.max_new_tokens == 0:
            return generated[:, 1:]

        past_key_values: Optional[DecoderPastKeyValues] = None
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=device)

        for _ in range(generation_config.max_new_tokens):
            if generation_config.use_cache:
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
            next_token = sample_next_token(logits, generation_config)
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, generation_config.eos_token_id),
                next_token,
            )
            generated = torch.cat((generated, next_token), dim=1)
            finished = finished | (next_token.squeeze(1) == generation_config.eos_token_id)
            if finished.all():
                break

        return generated[:, 1:]  # Exclude the initial BOS token
