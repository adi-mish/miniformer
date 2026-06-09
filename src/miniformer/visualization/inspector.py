from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import torch


@dataclass(frozen=True)
class LayerTrace:
    name: str
    shape: tuple[int, ...]
    mean: float
    std: float
    norm: float


@dataclass(frozen=True)
class AttentionTrace:
    name: str
    shape: tuple[int, ...]
    entropy: float


@dataclass(frozen=True)
class TransformerTrace:
    output_shape: tuple[int, ...]
    layers: List[LayerTrace]
    attentions: List[AttentionTrace]


def _as_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    if hasattr(output, "_dec_out") and isinstance(output._dec_out, torch.Tensor):
        return output._dec_out
    return None


def _layer_trace(name: str, tensor: torch.Tensor) -> LayerTrace:
    values = tensor.detach().float()
    return LayerTrace(
        name=name,
        shape=tuple(values.shape),
        mean=float(values.mean().cpu()),
        std=float(values.std(unbiased=False).cpu()),
        norm=float(values.norm().cpu()),
    )


def _attention_trace(name: str, tensor: torch.Tensor) -> AttentionTrace:
    values = tensor.detach().float().clamp_min(1e-12)
    entropy = -(values * values.log()).sum(dim=-1).mean()
    return AttentionTrace(
        name=name,
        shape=tuple(values.shape),
        entropy=float(entropy.cpu()),
    )


def _collect_attention_traces(model: torch.nn.Module) -> List[AttentionTrace]:
    traces: List[AttentionTrace] = []

    encoder = getattr(model, "encoder", None)
    for idx, attn in enumerate(getattr(encoder, "attn_weights", []) or []):
        if isinstance(attn, torch.Tensor):
            traces.append(_attention_trace(f"encoder.layers.{idx}.self_attention", attn))

    decoder = getattr(model, "decoder", None)
    for idx, attn in enumerate(getattr(decoder, "self_attentions", []) or []):
        if isinstance(attn, torch.Tensor):
            traces.append(_attention_trace(f"decoder.layers.{idx}.self_attention", attn))
    for idx, attn in enumerate(getattr(decoder, "cross_attentions", []) or []):
        if isinstance(attn, torch.Tensor):
            traces.append(_attention_trace(f"decoder.layers.{idx}.cross_attention", attn))

    return traces


def capture_transformer_trace(
    model: torch.nn.Module,
    *model_args,
    layer_name_fragments: Sequence[str] = (".layers.",),
    **model_kwargs,
) -> TransformerTrace:
    """Run one forward pass and capture layer/attention summary statistics."""
    layers: List[LayerTrace] = []
    hooks = []
    was_training = model.training

    def should_trace(name: str) -> bool:
        return (
            any(fragment in name for fragment in layer_name_fragments)
            and name.rsplit(".", 1)[-1].isdigit()
        )

    for name, module in model.named_modules():
        if should_trace(name):
            hooks.append(
                module.register_forward_hook(
                    lambda _m, _i, output, name=name: _record(name, output)
                )
            )

    def cleanup() -> None:
        for hook in hooks:
            hook.remove()

    def _record(name: str, output: Any) -> None:
        tensor = _as_tensor(output)
        if tensor is not None:
            layers.append(_layer_trace(name, tensor))

    try:
        model.eval()
        with torch.no_grad():
            output = model(*model_args, **model_kwargs)
    finally:
        cleanup()
        model.train(was_training)

    output_tensor = _as_tensor(output)
    if output_tensor is None:
        raise TypeError("Could not infer tensor output from model forward pass")

    return TransformerTrace(
        output_shape=tuple(output_tensor.shape),
        layers=layers,
        attentions=_collect_attention_traces(model),
    )


def plot_trace_summary(trace: TransformerTrace):
    """Plot per-layer activation norms from a captured trace."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(trace.layers) * 1.4), 4))
    names = [layer.name for layer in trace.layers]
    norms = [layer.norm for layer in trace.layers]
    ax.bar(range(len(norms)), norms)
    ax.set_xticks(range(len(norms)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Activation norm")
    ax.set_title("Transformer Layer Trace")
    fig.tight_layout()
    return fig, ax
