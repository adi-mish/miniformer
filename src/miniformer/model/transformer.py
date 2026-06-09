from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from miniformer.config.model_config import TransformerConfig
from miniformer.model.cache import EncoderPastKeyValues
from miniformer.model.encoder import Encoder
from miniformer.model.initialization import init_transformer_module
from miniformer.model.masks import causal_mask, padding_mask, self_attention_mask
from miniformer.model.outputs import TransformerModelOutput
from miniformer.utils.tokenization import stable_token_id

if TYPE_CHECKING:
    from miniformer.inspect import TransformerTrace


class Transformer(nn.Module):

    def __init__(self, config: Optional[TransformerConfig] = None, **kwargs):
        """
        Build an encoder-only Transformer with an explicit output mode.
        """
        super().__init__()

        # ── resolve / patch configuration ─────────────────────────────────
        if config is None:
            config = TransformerConfig(**kwargs) if kwargs else TransformerConfig()
        else:  # allow keyword overrides on an existing config
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        self.config = config
        self.pad_id = 0  # default padding token id

        # ── real encoder backbone ─────────────────────────────────────────
        self.encoder = Encoder(config)
        self.token_embedding = self.encoder.token_embedding  # expose for tests
        self.input_projection = self.encoder.input_projection  # expose for tests
        self.output_projection: nn.Module

        if config.output_mode == "hidden":
            self.output_projection = nn.Identity()
            self._tied_weights = False
        elif config.output_mode == "vocab":
            if self.token_embedding is None:
                raise ValueError("output_mode='vocab' requires token embeddings")
            self._tied_weights = True
            self.output_projection = nn.Identity()
        elif config.output_mode == "projection":
            assert config.output_dim is not None
            self.output_projection = nn.Linear(config.d_model, config.output_dim)
            self.output_projection.apply(
                lambda module: init_transformer_module(module, config.initializer_range)
            )
            self._tied_weights = False
        else:
            raise ValueError(f"Unknown output_mode: {config.output_mode}")

    def _build_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Build a padding mask, with optional autoregressive masking."""
        return self_attention_mask(seq, causal=self.config.causal, pad_id=self.pad_id)

    def forward(
        self,
        x: Union[torch.Tensor, List[Dict[str, Any]], Dict[str, torch.Tensor]],
        mask: Optional[torch.Tensor] = None,
        *,  # make the cache args keyword-only
        past_key_values: Optional[EncoderPastKeyValues] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> TransformerModelOutput:
        """Run the encoder-only transformer with optional causal KV caching."""
        # Handle non-tensor inputs (batches that are lists/dicts)
        if not isinstance(x, torch.Tensor):
            # For raw batch inputs during inference
            if isinstance(x, (list, tuple)) and isinstance(x[0], dict):
                # Simple feature extraction - just get the first element if it's a batch
                if "input_ids" in x[0]:
                    x = torch.stack([item["input_ids"] for item in x])
                elif "input" in x[0]:
                    # Deterministic lightweight token IDs for inference without a tokenizer.
                    vocab_size = self.config.vocab_size
                    texts = [item["input"] for item in x]
                    max_len = max(len(str(t).split()) for t in texts)
                    x = torch.zeros(
                        len(texts), max_len, dtype=torch.long, device=next(self.parameters()).device
                    )
                    for i, t in enumerate(texts):
                        words = str(t).split()
                        for j, w in enumerate(words):
                            x[i, j] = stable_token_id(w, vocab_size)
                else:
                    raise TypeError("Input batch dict must contain 'input_ids' or 'input' keys")
            elif isinstance(x, dict) and "input_ids" in x:
                # Handle dictionary with input_ids directly
                x = x["input_ids"]
            else:
                raise TypeError(
                    "Input must be a tensor, a list of dicts with "
                    "'input_ids' or 'input', or a dict with 'input_ids'"
                )

        past_len = 0
        if use_cache:
            if not self.config.causal:
                raise RuntimeError("KV caching requires causal=True")
            if mask is not None:
                raise ValueError("custom masks are not supported with cached encoder decoding")
            if x.dim() == 2 and x.dtype == torch.long and (x == self.pad_id).any():
                raise ValueError("cached encoder decoding does not support padding tokens")
            if past_key_values is not None and len(past_key_values) > 0:
                first_past = past_key_values[0]
                if first_past is not None:
                    past_len = first_past.key.size(2)

        # ----- build / reuse the attention mask -----------------------------
        if mask is None:
            if use_cache:
                mask = causal_mask(x.size(1), past_len=past_len, device=x.device)
            else:
                mask = self._build_mask(x)

        # ----- run encoder --------------------------------------------------
        if self.encoder is None:
            raise RuntimeError(
                "Encoder is not initialized properly. Check the Encoder class and configuration."
            )
        encoder_output = self.encoder(
            x,
            mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        h_full = encoder_output.hidden_states  # [B, S, d_model]

        # ----- explicit output head -----------------------------------------
        if getattr(self, "_tied_weights", False) and self.token_embedding is not None:
            out_full = torch.matmul(h_full, self.token_embedding.weight.t())
        else:
            out_full = self.output_projection(h_full)

        return TransformerModelOutput(
            logits=out_full if self.config.output_mode == "vocab" else None,
            hidden_states=out_full if self.config.output_mode == "hidden" else None,
            projection=out_full if self.config.output_mode == "projection" else None,
            self_attentions=encoder_output.self_attentions,
            past_key_values=encoder_output.past_key_values,
        )

    def _create_mask(self, x):
        """Create a mask to hide padding tokens"""
        return padding_mask(x, pad_id=self.pad_id)

    def get_attention_weights(self, x):
        """Get attention weights for visualization"""
        mask = self._build_mask(x)
        _ = self.forward(x, mask)
        # after __init__ is done, encoder is never None
        assert self.encoder is not None, "Transformer.encoder should already be initialized"
        return self.encoder.attn_weights

    def trace(
        self,
        x: Union[torch.Tensor, List[Dict[str, Any]], Dict[str, torch.Tensor]],
        mask: Optional[torch.Tensor] = None,
        *,
        top_k: int = 5,
        compare_cache: bool = False,
        **kwargs,
    ) -> TransformerTrace:
        """Capture a structured inspection trace for one encoder-only forward pass."""
        from miniformer.inspect import capture_transformer_trace

        return capture_transformer_trace(
            self,
            x,
            mask=mask,
            top_k=top_k,
            compare_cache=compare_cache,
            **kwargs,
        )

    def save_pretrained(self, save_dir: str) -> None:
        """Save model and configuration to directory"""
        os.makedirs(save_dir, exist_ok=True)

        # Save model weights
        model_path = os.path.join(save_dir, "model.pt")
        torch.save(self.state_dict(), model_path)

        # Save configuration
        config_path = os.path.join(save_dir, "config.json")
        self.config.save_json(config_path)

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "Transformer":
        """Load model from directory"""
        config_path = os.path.join(model_dir, "config.json")
        config = TransformerConfig.from_json(config_path)

        model = cls(config)
        model_path = os.path.join(model_dir, "model.pt")
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        return model
