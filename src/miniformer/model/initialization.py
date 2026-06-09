from __future__ import annotations

import torch.nn as nn


def init_transformer_module(module: nn.Module, initializer_range: float) -> None:
    """Initialize standard transformer modules from one config-controlled policy."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
