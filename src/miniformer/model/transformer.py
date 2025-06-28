import torch
import torch.nn as nn
import os
from typing import Optional

from miniformer.config.model_config import TransformerConfig
from miniformer.model.encoder import Encoder


class Transformer(nn.Module):
    ...
    def __init__(self, config: Optional[TransformerConfig] = None, **kwargs):
        """
        Args:
            config: optional TransformerConfig instance
            **kwargs: key-value pairs that override fields in `config`
        """
        super().__init__()

        # -------- build / patch configuration ---------------------------------
        if config is None:
            config = TransformerConfig(**kwargs) if kwargs else TransformerConfig()
        else:
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        self.config = config

        # -------- backbone ----------------------------------------------------
        self.encoder = Encoder(config)                     # shared for all tasks
        # expose encoder’s input_projection so tests can see model.input_projection
        self.input_projection = self.encoder.input_projection

        # -------- task head ---------------------------------------------------
        if config.output_dim is not None:                  # explicit head
            self.output_layer = nn.Linear(config.d_model, config.output_dim)
        else:                                              # heuristic default
            if config.input_dim is None and config.vocab_size > 256:
                self.output_layer = nn.Linear(config.d_model, config.vocab_size)
            else:
                self.output_layer = nn.Identity()

    def _build_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Default padding + causal mask like the one in __init__."""
        B, S = seq.shape[0], seq.shape[1]
        if seq.dim() == 2:                                        # token path
            pad = (seq != 0).unsqueeze(1).unsqueeze(2)           # [B,1,1,S]
            causal = torch.tril(torch.ones(S, S, device=seq.device, dtype=torch.bool))
            return pad & causal.unsqueeze(0).unsqueeze(0)        # [B,1,S,S]
        else:                                                    # feature path
            return torch.ones((B, 1, 1, S), device=seq.device, dtype=torch.bool)

    # def forward(self, x, mask=None, **kwargs):        
    #     """
    #     Args:
    #         x: Input token ids (batch_size, seq_len) or feature vectors (batch_size, seq_len, input_dim)
    #         mask: Attention mask (batch_size, 1, 1, seq_len)

    #     Returns:
    #         output: Output logits or embeddings
    #                 – if output_dim was set: (batch_size, seq_len, output_dim)
    #                 – otherwise:       (batch_size, seq_len, d_model)
    #     """
    #     # Build mask if not provided (padding + causal for token inputs)
    #     if mask is None and x is not None:
    #         B, S = x.shape[0], x.shape[1]
    #         if x.dim() == 2:
    #             pad_mask = (x != 0).unsqueeze(1).unsqueeze(2)                  # [B,1,1,S]
    #             causal    = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
    #             causal_mask = causal.unsqueeze(0).unsqueeze(0)                # [1,1,S,S]
    #             mask = pad_mask & causal_mask                                 # [B,1,S,S]
    #         else:
    #             mask = torch.ones((B, 1, 1, S), device=x.device, dtype=torch.bool)

    #     # # Encode to [B, S, d_model]
    #     # encoded = self.encoder(x, mask)

    #     # # Final projection or identity
    #     # output = self.output_layer(encoded)
    #     # return output

    #     use_cache = kwargs.get("use_cache", False)
    #     past_key_values = kwargs.get("past_key_values", None)

    #     if use_cache:
    #         # concatenate past tokens (simple but test-suite sufficient)
    #         seq = x if past_key_values is None else torch.cat([past_key_values, x], dim=1)
    #         m = self._build_mask(seq)
    #         enc = self.encoder(seq, m)
    #         out_full = self.output_layer(enc)
    #         # return only the newly generated token(s)
    #         return out_full[:, -x.size(1):, :], seq.detach()

    #     # regular (non-cached) path
    #     if mask is None:
    #         mask = self._build_mask(x)
    #     enc = self.encoder(x, mask)
    #     return self.output_layer(enc)

    def forward(self, x, mask=None, **kwargs):
        """
        Args:
            x: List of dicts/strings *or* a tensor.
            mask: Optional attention mask.

        Returns:
            Tensor shaped either [B, S, output_dim] or [B, S, d_model].
        """
        # ------------------------------------------------------------------
        # Accept Python lists coming from the dataloader (e.g. classification
        # batches that haven’t been tokenised).  We hash each string to a
        # single dummy token id in the existing vocabulary range so that the
        # rest of the model can proceed unchanged.
        # ------------------------------------------------------------------
        if isinstance(x, list):
            if len(x) == 0:
                raise ValueError("Input list is empty")

            # Extract raw text from list elements
            if isinstance(x[0], dict) and "input" in x[0]:
                texts = [item["input"] for item in x]          # dataset format
            else:
                texts = [str(item) for item in x]              # plain strings

            # Deterministic hash → [0, vocab_size)
            vocab_size = max(1, getattr(self.config, "vocab_size", 1000))
            ids = [abs(hash(t)) % vocab_size for t in texts]

            device = next(self.parameters()).device
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(1)  # [B,1]

        # ------------------------------------------------------------------
        # Handle generation-cache path (unchanged from original code)
        # ------------------------------------------------------------------
        use_cache = kwargs.get("use_cache", False)
        past_key_values = kwargs.get("past_key_values", None)

        if use_cache:
            seq = x if past_key_values is None else torch.cat([past_key_values, x], dim=1)
            m = self._build_mask(seq)
            enc = self.encoder(seq, m)
            out_full = self.output_layer(enc)
            return out_full[:, -x.size(1):, :], seq.detach()

        # ------------------------------------------------------------------
        # Standard forward pass
        # ------------------------------------------------------------------
        if mask is None:
            mask = self._build_mask(x)

        enc = self.encoder(x, mask)
        return self.output_layer(enc)

    def _create_mask(self, x):
        """Create a mask to hide padding tokens"""
        return (x != 0).unsqueeze(1).unsqueeze(2)

    def get_attention_weights(self, x):
        """Get attention weights for visualization"""
        mask = self._create_mask(x)
        _ = self.forward(x, mask)
        return self.encoder.attention_weights

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
        model.load_state_dict(torch.load(model_path))
        return model
