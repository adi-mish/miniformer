from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence, cast

import torch
import torch.nn.functional as F

from miniformer.model.masks import padding_mask


@dataclass(frozen=True)
class TensorSummary:
    """Small JSON-serializable summary of a tensor."""

    name: str
    shape: tuple[int, ...]
    mean: float
    std: float
    minimum: float
    maximum: float
    norm: float


@dataclass(frozen=True)
class AttentionTrace:
    """Summary of one attention module from a forward pass."""

    name: str
    shape: tuple[int, ...]
    entropy: Optional[float]
    max_probability: Optional[float]
    mean_probability: Optional[float]
    available: bool = True
    reason: Optional[str] = None


@dataclass(frozen=True)
class LayerTrace:
    """Per-transformer-layer view of residual, attention, and MLP behavior."""

    name: str
    shape: tuple[int, ...]
    mean: float
    std: float
    norm: float
    input_norm: float
    output_norm: float
    residual_delta_norm: Optional[float]
    self_attention_output_norm: Optional[float] = None
    cross_attention_output_norm: Optional[float] = None
    mlp_activation_norm: Optional[float] = None
    mlp_output_norm: Optional[float] = None


@dataclass(frozen=True)
class LogitTrace:
    """Top-k predictions for every output position when the output is logit-like."""

    shape: tuple[int, ...]
    top_k: int
    token_ids: List[List[List[int]]]
    values: List[List[List[float]]]
    probabilities: List[List[List[float]]]


@dataclass(frozen=True)
class CacheTrace:
    """Result of comparing full-sequence output with cached step-by-step output."""

    attempted: bool
    supported: bool
    allclose: Optional[bool] = None
    max_abs_diff: Optional[float] = None
    tokens_compared: int = 0
    atol: float = 1e-5
    rtol: float = 1e-5
    reason: Optional[str] = None


@dataclass(frozen=True)
class TransformerTrace:
    """Complete structured trace for one model forward pass."""

    output_shape: tuple[int, ...]
    layers: List[LayerTrace]
    attentions: List[AttentionTrace]
    logits: Optional[LogitTrace] = None
    cache: CacheTrace = field(
        default_factory=lambda: CacheTrace(
            attempted=False,
            supported=False,
            reason="cache comparison disabled",
        )
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialize the trace to JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path, *, indent: Optional[int] = 2) -> None:
        """Write the trace to a JSON file."""
        Path(path).write_text(self.to_json(indent=indent))


@dataclass
class _LayerState:
    input: Optional[TensorSummary] = None
    input_tensor: Optional[torch.Tensor] = None
    output: Optional[TensorSummary] = None
    output_tensor: Optional[torch.Tensor] = None
    self_attention_output_norm: Optional[float] = None
    cross_attention_output_norm: Optional[float] = None
    mlp_activation_norm: Optional[float] = None
    mlp_output_norm: Optional[float] = None


def _as_tensor(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "output") and isinstance(output.output, torch.Tensor):
        return output.output
    if hasattr(output, "logits") and isinstance(output.logits, torch.Tensor):
        return output.logits
    if hasattr(output, "hidden_states") and isinstance(output.hidden_states, torch.Tensor):
        return output.hidden_states
    return None


def _summary(name: str, tensor: torch.Tensor) -> TensorSummary:
    values = tensor.detach().float()
    return TensorSummary(
        name=name,
        shape=tuple(values.shape),
        mean=float(values.mean().cpu()),
        std=float(values.std(unbiased=False).cpu()),
        minimum=float(values.min().cpu()),
        maximum=float(values.max().cpu()),
        norm=float(values.norm().cpu()),
    )


def _attention_trace(name: str, tensor: Optional[torch.Tensor]) -> AttentionTrace:
    if tensor is None:
        return AttentionTrace(
            name=name,
            shape=(),
            entropy=None,
            max_probability=None,
            mean_probability=None,
            available=False,
            reason="attention weights unavailable; use_sdpa=True does not return weights",
        )

    values = tensor.detach().float()
    clamped = values.clamp_min(1e-12)
    entropy = -(clamped * clamped.log()).sum(dim=-1).mean()
    return AttentionTrace(
        name=name,
        shape=tuple(values.shape),
        entropy=float(entropy.cpu()),
        max_probability=float(values.max().cpu()),
        mean_probability=float(values.mean().cpu()),
    )


def _is_layer_name(name: str, fragments: Sequence[str]) -> bool:
    return any(fragment in name for fragment in fragments) and name.rsplit(".", 1)[-1].isdigit()


def _layer_name_from_child(name: str, suffix: str) -> str:
    return name[: -len(suffix)]


def _safe_norm(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().norm().cpu())


def _feed_forward_activation(module: torch.nn.Module, x: torch.Tensor) -> Optional[torch.Tensor]:
    activation_name = getattr(module, "activation_name", None)
    feed_forward = cast(Any, module)
    if activation_name == "swiglu" and hasattr(module, "w12"):
        x1, x2 = feed_forward.w12(x).chunk(2, dim=-1)
        return F.silu(x1) * x2
    if activation_name in {"gelu", "relu"} and hasattr(module, "linear1"):
        hidden = feed_forward.linear1(x)
        return F.gelu(hidden) if activation_name == "gelu" else F.relu(hidden)
    return None


def _is_logit_like(model: torch.nn.Module, output: torch.Tensor) -> bool:
    if output.dim() < 2:
        return False
    config = getattr(model, "config", None)
    if config is None:
        return True
    output_mode = getattr(config, "output_mode", None)
    if output_mode in {"vocab", "projection"}:
        return True
    if output_mode == "hidden":
        return False
    return False


def _logit_trace(output: torch.Tensor, top_k: int) -> Optional[LogitTrace]:
    if top_k <= 0 or output.dim() < 2:
        return None

    values = output.detach().float()
    k = min(top_k, values.size(-1))
    top_values, top_indices = torch.topk(values, k=k, dim=-1)
    top_probabilities = torch.gather(values.softmax(dim=-1), dim=-1, index=top_indices)
    return LogitTrace(
        shape=tuple(values.shape),
        top_k=k,
        token_ids=top_indices.cpu().tolist(),
        values=top_values.cpu().tolist(),
        probabilities=top_probabilities.cpu().tolist(),
    )


def _compare_encoder_cache(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    model_kwargs: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
) -> CacheTrace:
    if input_ids.dim() != 2 or input_ids.dtype != torch.long:
        return CacheTrace(
            attempted=True,
            supported=False,
            atol=atol,
            rtol=rtol,
            reason="encoder cache comparison requires a [batch, seq] integer token tensor",
        )
    if getattr(model, "token_embedding", None) is None:
        return CacheTrace(
            attempted=True,
            supported=False,
            atol=atol,
            rtol=rtol,
            reason="encoder cache comparison is only available for token models",
        )
    if model_kwargs.get("mask") is not None:
        return CacheTrace(
            attempted=True,
            supported=False,
            atol=atol,
            rtol=rtol,
            reason="custom masks are not replayed by the cache comparison probe",
        )

    full = model(input_ids, use_cache=False)
    full_tensor = _as_tensor(full)
    if full_tensor is None:
        return CacheTrace(
            attempted=True,
            supported=False,
            atol=atol,
            rtol=rtol,
            reason="could not infer full-sequence tensor output",
        )

    past_key_values = None
    cached_outputs: list[torch.Tensor] = []
    for index in range(input_ids.size(1)):
        current = input_ids[:, index : index + 1]
        output = model(
            current,
            past_key_values=past_key_values,
            use_cache=True,
        )
        cached_outputs.append(output.output)
        past_key_values = output.past_key_values

    cached = torch.cat(cached_outputs, dim=1)
    diff = (full_tensor - cached).abs().max()
    return CacheTrace(
        attempted=True,
        supported=True,
        allclose=bool(torch.allclose(full_tensor, cached, atol=atol, rtol=rtol)),
        max_abs_diff=float(diff.cpu()),
        tokens_compared=input_ids.size(1),
        atol=atol,
        rtol=rtol,
    )


def _compare_seq2seq_cache(
    model: torch.nn.Module,
    src: torch.Tensor,
    tgt: torch.Tensor,
    model_kwargs: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
) -> CacheTrace:
    if tgt.dim() != 2 or tgt.dtype != torch.long:
        return CacheTrace(
            attempted=True,
            supported=False,
            atol=atol,
            rtol=rtol,
            reason="seq2seq cache comparison requires an integer target token tensor",
        )
    if (tgt == 0).any():
        return CacheTrace(
            attempted=True,
            supported=False,
            atol=atol,
            rtol=rtol,
            reason="target padding is not replayed by the cache comparison probe",
        )
    if any(model_kwargs.get(key) is not None for key in ("src_mask", "tgt_mask", "memory_mask")):
        return CacheTrace(
            attempted=True,
            supported=False,
            atol=atol,
            rtol=rtol,
            reason="custom masks are not replayed by the cache comparison probe",
        )
    if not model_kwargs.get("use_causal_mask", True):
        return CacheTrace(
            attempted=True,
            supported=False,
            atol=atol,
            rtol=rtol,
            reason="cache comparison only applies to causal decoder passes",
        )

    full = model(src, tgt)
    full_tensor = _as_tensor(full)
    if full_tensor is None:
        return CacheTrace(
            attempted=True,
            supported=False,
            atol=atol,
            rtol=rtol,
            reason="could not infer full-sequence tensor output",
        )

    src_mask = padding_mask(src)
    model_any = cast(Any, model)
    enc_input_proj = getattr(model_any, "_enc_input_proj", None)
    src_proj = enc_input_proj(src) if enc_input_proj is not None and src.dim() == 3 else src
    memory = model_any.encoder(src_proj, src_mask)

    past_key_values = None
    cached_outputs: list[torch.Tensor] = []
    for index in range(tgt.size(1)):
        current = tgt[:, index : index + 1]
        decoder_output = model_any.decoder(
            current,
            memory,
            None,
            src_mask,
            use_causal_mask=True,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = decoder_output.past_key_values
        cached_outputs.append(decoder_output.output)

    cached = torch.cat(cached_outputs, dim=1)
    diff = (full_tensor - cached).abs().max()
    return CacheTrace(
        attempted=True,
        supported=True,
        allclose=bool(torch.allclose(full_tensor, cached, atol=atol, rtol=rtol)),
        max_abs_diff=float(diff.cpu()),
        tokens_compared=tgt.size(1),
        atol=atol,
        rtol=rtol,
    )


def _compare_cache(
    model: torch.nn.Module,
    model_args: tuple[Any, ...],
    model_kwargs: Mapping[str, Any],
    *,
    compare_cache: bool,
    atol: float,
    rtol: float,
) -> CacheTrace:
    if not compare_cache:
        return CacheTrace(
            attempted=False,
            supported=False,
            atol=atol,
            rtol=rtol,
            reason="cache comparison disabled",
        )

    if (
        len(model_args) == 1
        and isinstance(model_args[0], torch.Tensor)
        and hasattr(model, "encoder")
    ):
        return _compare_encoder_cache(model, model_args[0], model_kwargs, atol=atol, rtol=rtol)
    if (
        len(model_args) >= 2
        and isinstance(model_args[0], torch.Tensor)
        and isinstance(model_args[1], torch.Tensor)
        and hasattr(model, "decoder")
    ):
        return _compare_seq2seq_cache(
            model,
            model_args[0],
            model_args[1],
            model_kwargs,
            atol=atol,
            rtol=rtol,
        )

    return CacheTrace(
        attempted=True,
        supported=False,
        atol=atol,
        rtol=rtol,
        reason="model signature is not recognized by the cache comparison probe",
    )


def _build_layer_traces(
    layer_order: Sequence[str],
    layer_states: Mapping[str, _LayerState],
) -> List[LayerTrace]:
    layers: list[LayerTrace] = []
    for name in layer_order:
        state = layer_states[name]
        if state.output is None:
            continue
        residual_delta_norm = None
        input_norm = state.output.norm
        if state.input is not None:
            input_norm = state.input.norm
        if (
            state.input_tensor is not None
            and state.output_tensor is not None
            and state.input_tensor.shape == state.output_tensor.shape
        ):
            residual_delta_norm = _safe_norm(state.output_tensor - state.input_tensor)
        layers.append(
            LayerTrace(
                name=name,
                shape=state.output.shape,
                mean=state.output.mean,
                std=state.output.std,
                norm=state.output.norm,
                input_norm=input_norm,
                output_norm=state.output.norm,
                residual_delta_norm=residual_delta_norm,
                self_attention_output_norm=state.self_attention_output_norm,
                cross_attention_output_norm=state.cross_attention_output_norm,
                mlp_activation_norm=state.mlp_activation_norm,
                mlp_output_norm=state.mlp_output_norm,
            )
        )
    return layers


def capture_transformer_trace(
    model: torch.nn.Module,
    *model_args: Any,
    layer_name_fragments: Sequence[str] = (".layers.",),
    top_k: int = 5,
    compare_cache: bool = False,
    cache_atol: float = 1e-5,
    cache_rtol: float = 1e-5,
    **model_kwargs: Any,
) -> TransformerTrace:
    """
    Run one eval/no-grad forward pass and return a structured transformer trace.

    The tracer uses temporary forward hooks and always removes them before
    returning or re-raising an exception. The model's training/eval mode is
    restored afterward.
    """
    layer_states: MutableMapping[str, _LayerState] = {}
    layer_order: list[str] = []
    attentions: list[AttentionTrace] = []
    hooks: list[Any] = []
    was_training = model.training
    output: Any = None

    def ensure_layer(name: str) -> _LayerState:
        if name not in layer_states:
            layer_states[name] = _LayerState()
            layer_order.append(name)
        return layer_states[name]

    def layer_pre_hook(name: str, _module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
        tensor = _as_tensor(inputs[0]) if inputs else None
        if tensor is not None:
            state = ensure_layer(name)
            state.input = _summary(f"{name}.input", tensor)
            state.input_tensor = tensor.detach()

    def layer_hook(name: str, _module: torch.nn.Module, _inputs: tuple[Any, ...], out: Any) -> None:
        tensor = _as_tensor(out)
        if tensor is not None:
            state = ensure_layer(name)
            state.output = _summary(name, tensor)
            state.output_tensor = tensor.detach()

    def attention_hook(
        name: str,
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        out: Any,
    ) -> None:
        tensor = _as_tensor(out)
        if tensor is None and isinstance(out, (tuple, list)) and out:
            first = out[0]
            tensor = first if isinstance(first, torch.Tensor) else None
        layer_name = name.rsplit(".", 1)[0]
        state = ensure_layer(layer_name)
        if tensor is not None:
            if name.endswith(".self_attention"):
                state.self_attention_output_norm = _safe_norm(tensor)
            elif name.endswith(".cross_attention"):
                state.cross_attention_output_norm = _safe_norm(tensor)
        attn = None
        if isinstance(out, (tuple, list)) and len(out) > 1:
            candidate = out[1]
            attn = candidate if isinstance(candidate, torch.Tensor) else None
        attentions.append(_attention_trace(name, attn))

    def feed_forward_hook(
        name: str,
        module: torch.nn.Module,
        inputs: tuple[Any, ...],
        out: Any,
    ) -> None:
        layer_name = _layer_name_from_child(name, ".feed_forward")
        state = ensure_layer(layer_name)
        tensor = _as_tensor(out)
        if tensor is not None:
            state.mlp_output_norm = _safe_norm(tensor)
        if inputs and isinstance(inputs[0], torch.Tensor):
            activation = _feed_forward_activation(module, inputs[0])
            if activation is not None:
                state.mlp_activation_norm = _safe_norm(activation)

    for name, module in model.named_modules():
        if _is_layer_name(name, layer_name_fragments):
            ensure_layer(name)
            hooks.append(
                module.register_forward_pre_hook(
                    lambda module, inputs, name=name: layer_pre_hook(name, module, inputs)
                )
            )
            hooks.append(
                module.register_forward_hook(
                    lambda module, inputs, out, name=name: layer_hook(name, module, inputs, out)
                )
            )
        elif name.endswith(".self_attention") or name.endswith(".cross_attention"):
            hooks.append(
                module.register_forward_hook(
                    lambda module, inputs, out, name=name: attention_hook(
                        name,
                        module,
                        inputs,
                        out,
                    )
                )
            )
        elif name.endswith(".feed_forward"):
            hooks.append(
                module.register_forward_hook(
                    lambda module, inputs, out, name=name: feed_forward_hook(
                        name,
                        module,
                        inputs,
                        out,
                    )
                )
            )

    def cleanup() -> None:
        for hook in hooks:
            hook.remove()

    try:
        model.eval()
        with torch.no_grad():
            output = model(*model_args, **model_kwargs)
    except Exception:
        cleanup()
        model.train(was_training)
        raise
    cleanup()

    output_tensor = _as_tensor(output)
    if output_tensor is None:
        model.train(was_training)
        raise TypeError("Could not infer tensor output from model forward pass")

    try:
        with torch.no_grad():
            cache = _compare_cache(
                model,
                model_args,
                model_kwargs,
                compare_cache=compare_cache,
                atol=cache_atol,
                rtol=cache_rtol,
            )
    finally:
        model.train(was_training)

    logits = _logit_trace(output_tensor, top_k) if _is_logit_like(model, output_tensor) else None
    return TransformerTrace(
        output_shape=tuple(output_tensor.shape),
        layers=_build_layer_traces(layer_order, layer_states),
        attentions=attentions,
        logits=logits,
        cache=cache,
        metadata={
            "model_class": model.__class__.__name__,
            "training_restored_to": was_training,
        },
    )


def plot_trace_summary(trace: TransformerTrace):
    """Plot residual, attention, and MLP norms from a captured trace."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(7, len(trace.layers) * 1.5), 4.5))
    names = [layer.name for layer in trace.layers]
    positions = list(range(len(trace.layers)))
    output_norms = [layer.output_norm for layer in trace.layers]
    mlp_norms = [layer.mlp_output_norm or 0.0 for layer in trace.layers]
    attention_norms = [layer.self_attention_output_norm or 0.0 for layer in trace.layers]

    width = 0.25
    ax.bar([position - width for position in positions], output_norms, width, label="residual")
    ax.bar(positions, attention_norms, width, label="attention")
    ax.bar([position + width for position in positions], mlp_norms, width, label="mlp")
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Activation norm")
    ax.set_title("Transformer Trace")
    ax.legend()
    fig.tight_layout()
    return fig, ax
