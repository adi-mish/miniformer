from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence, cast

import torch
import torch.nn.functional as F

from miniformer.model.masks import padding_mask

DEFAULT_MAX_REPORT_TOKENS = 64
DEFAULT_MAX_REPORT_HEADS = 8


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
    per_head_entropy: Optional[List[float]]
    max_probability: Optional[float]
    mean_probability: Optional[float]
    q_projection: Optional[TensorSummary] = None
    k_projection: Optional[TensorSummary] = None
    v_projection: Optional[TensorSummary] = None
    weights: Optional[List[List[List[float]]]] = None
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
    residual_by_token: Optional[List[List[float]]] = None
    self_attention_output_norm: Optional[float] = None
    cross_attention_output_norm: Optional[float] = None
    mlp_activation_norm: Optional[float] = None
    mlp_output_norm: Optional[float] = None
    mlp_activation: Optional[TensorSummary] = None
    mlp_output: Optional[TensorSummary] = None


@dataclass(frozen=True)
class LogitTrace:
    """Top-k predictions for every output position when the output is logit-like."""

    shape: tuple[int, ...]
    top_k: int
    token_ids: List[List[List[int]]]
    values: List[List[List[float]]]
    probabilities: List[List[List[float]]]
    entropy: List[List[float]]
    top_token_ids: List[List[int]]


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

    def to_html(
        self,
        path: str | Path,
        tokens: Optional[Sequence[str]] = None,
        *,
        max_tokens: int = DEFAULT_MAX_REPORT_TOKENS,
        max_heads: int = DEFAULT_MAX_REPORT_HEADS,
    ) -> None:
        """Write a self-contained HTML report for this trace."""
        save_trace_html(self, path, tokens=tokens, max_tokens=max_tokens, max_heads=max_heads)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def _shape_text(shape: Sequence[int]) -> str:
    return "x".join(str(item) for item in shape) if shape else ""


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _matrix_shape(values: Sequence[Sequence[float]]) -> tuple[int, int]:
    if not values:
        return (0, 0)
    return (len(values), max(len(row) for row in values))


def _heatmap_table(
    values: Sequence[Sequence[float]],
    *,
    tokens: Optional[Sequence[str]] = None,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
) -> str:
    if not values:
        return "<p>No values.</p>"
    row_count, col_count = _matrix_shape(values)
    display_values = values
    notice = ""
    if max_rows is not None or max_cols is not None:
        row_limit = max_rows if max_rows is not None else row_count
        col_limit = max_cols if max_cols is not None else col_count
        if row_count > row_limit or col_count > col_limit:
            display_values = [row[:col_limit] for row in values[:row_limit]]
            notice = (
                "<p class='notice'>Heatmap truncated from "
                f"{row_count}x{col_count} to {len(display_values)}x"
                f"{_matrix_shape(display_values)[1]} by report limits.</p>"
            )

    flat = [float(item) for row in display_values for item in row]
    minimum = min(flat)
    maximum = max(flat)
    span = max(maximum - minimum, 1e-12)
    header = "<tr><th></th>"
    width = max(len(row) for row in display_values)
    for col in range(width):
        label = tokens[col] if tokens is not None and col < len(tokens) else str(col)
        header += f"<th>{html.escape(label)}</th>"
    header += "</tr>"
    rows = [header]
    for row_index, row in enumerate(display_values):
        label = f"b{row_index}"
        cells = [f"<th>{html.escape(label)}</th>"]
        for value in row:
            intensity = (float(value) - minimum) / span
            color = int(255 - 155 * intensity)
            cells.append(
                "<td style='background:"
                f"rgb(255,{color},{color})' title='{html.escape(_fmt(float(value)))}'>"
                f"{html.escape(_fmt(float(value)))}</td>"
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return notice + "<table class='heatmap'>" + "".join(rows) + "</table>"


def _summary_row(label: str, summary: Optional[TensorSummary]) -> str:
    if summary is None:
        return f"<tr><td>{html.escape(label)}</td><td colspan='6'>Unavailable</td></tr>"
    return (
        f"<tr><td>{html.escape(label)}</td>"
        f"<td>{html.escape(_shape_text(summary.shape))}</td>"
        f"<td>{html.escape(_fmt(summary.mean))}</td>"
        f"<td>{html.escape(_fmt(summary.std))}</td>"
        f"<td>{html.escape(_fmt(summary.minimum))}</td>"
        f"<td>{html.escape(_fmt(summary.maximum))}</td>"
        f"<td>{html.escape(_fmt(summary.norm))}</td></tr>"
    )


def _summary_table(title: str, rows: Sequence[tuple[str, Optional[TensorSummary]]]) -> str:
    body = "".join(_summary_row(label, summary) for label, summary in rows)
    return (
        f"<h4>{html.escape(title)}</h4><table>"
        "<tr><th>Name</th><th>Shape</th><th>Mean</th><th>Std</th>"
        "<th>Min</th><th>Max</th><th>Norm</th></tr>"
        f"{body}</table>"
    )


def _attention_section(
    attention: AttentionTrace,
    tokens: Optional[Sequence[str]],
    *,
    max_tokens: int,
    max_heads: int,
) -> str:
    parts = [
        f"<section><h3>{html.escape(attention.name)}</h3>",
        "<p>"
        f"Shape: {html.escape(_shape_text(attention.shape))} | "
        f"Entropy: {html.escape(_fmt(attention.entropy))} | "
        f"Max probability: {html.escape(_fmt(attention.max_probability))}"
        "</p>",
        _summary_table(
            "Q/K/V Projection Summaries",
            [
                ("Q", attention.q_projection),
                ("K", attention.k_projection),
                ("V", attention.v_projection),
            ],
        ),
    ]
    if attention.per_head_entropy is not None:
        entropy_values = ", ".join(_fmt(value) for value in attention.per_head_entropy)
        parts.append(f"<p>Per-head entropy: {html.escape(entropy_values)}</p>")
    if not attention.available:
        parts.append(f"<p class='notice'>{html.escape(attention.reason or 'Unavailable')}</p>")
    elif attention.weights is not None:
        head_count = len(attention.weights)
        if head_count > max_heads:
            parts.append(
                "<p class='notice'>Raw attention heatmaps omitted by report head limit "
                f"({head_count} heads > {max_heads}).</p>"
            )
        else:
            for head_index, head_weights in enumerate(attention.weights):
                row_count, col_count = _matrix_shape(head_weights)
                if row_count > max_tokens or col_count > max_tokens:
                    parts.append(
                        "<p class='notice'>Raw attention heatmap omitted by report token limit "
                        f"({row_count}x{col_count} > {max_tokens}).</p>"
                    )
                    continue
                parts.append(f"<h4>Raw Attention Heatmap Head {head_index}</h4>")
                parts.append(
                    _heatmap_table(
                        head_weights,
                        tokens=tokens,
                        max_rows=max_tokens,
                        max_cols=max_tokens,
                    )
                )
    elif attention.reason:
        parts.append(f"<p class='notice'>{html.escape(attention.reason)}</p>")
    else:
        parts.append(
            "<p class='notice'>Raw attention heatmaps were not stored. Capture with "
            "include_raw_attention=True and use_sdpa=False to include them.</p>"
        )
    parts.append("</section>")
    return "".join(parts)


def _logit_section(
    logits: LogitTrace,
    tokens: Optional[Sequence[str]],
    *,
    max_tokens: int,
) -> str:
    rows = []
    first_batch_ids = logits.token_ids[0] if logits.token_ids else []
    first_batch_values = logits.values[0] if logits.values else []
    first_batch_probs = logits.probabilities[0] if logits.probabilities else []
    first_batch_entropy = logits.entropy[0] if logits.entropy else []
    truncated = len(first_batch_ids) > max_tokens
    for index, token_ids in enumerate(first_batch_ids[:max_tokens]):
        label = tokens[index] if tokens is not None and index < len(tokens) else str(index)
        values = first_batch_values[index]
        probs = first_batch_probs[index]
        top_items = ", ".join(
            f"{token_id}:{_fmt(value)} ({_fmt(prob)})"
            for token_id, value, prob in zip(token_ids, values, probs)
        )
        entropy = first_batch_entropy[index] if index < len(first_batch_entropy) else None
        rows.append(
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{html.escape(str(logits.top_token_ids[0][index]))}</td>"
            f"<td>{html.escape(_fmt(entropy))}</td>"
            f"<td>{html.escape(top_items)}</td></tr>"
        )
    notice = (
        f"<p class='notice'>Logit table truncated to {max_tokens} positions.</p>"
        if truncated
        else ""
    )
    return (
        "<section><h2>Logit Evolution</h2>"
        + notice
        + "<table><tr><th>Token</th><th>Top token</th><th>Entropy</th><th>Top-k</th></tr>"
        + "".join(rows)
        + "</table></section>"
    )


def save_trace_html(
    trace: TransformerTrace,
    path: str | Path,
    tokens: Optional[Sequence[str]] = None,
    *,
    max_tokens: int = DEFAULT_MAX_REPORT_TOKENS,
    max_heads: int = DEFAULT_MAX_REPORT_HEADS,
) -> None:
    """Write a self-contained static HTML report for a transformer trace."""
    _validate_positive_int("max_tokens", max_tokens)
    _validate_positive_int("max_heads", max_heads)
    cache = trace.cache
    cache_status = (
        f"attempted={cache.attempted}, supported={cache.supported}, "
        f"allclose={cache.allclose}, max_abs_diff={_fmt(cache.max_abs_diff)}, "
        f"tokens_compared={cache.tokens_compared}"
    )
    layer_rows = []
    residual_sections = []
    for layer in trace.layers:
        layer_rows.append(
            f"<tr><td>{html.escape(layer.name)}</td>"
            f"<td>{html.escape(_shape_text(layer.shape))}</td>"
            f"<td>{html.escape(_fmt(layer.input_norm))}</td>"
            f"<td>{html.escape(_fmt(layer.output_norm))}</td>"
            f"<td>{html.escape(_fmt(layer.residual_delta_norm))}</td>"
            f"<td>{html.escape(_fmt(layer.self_attention_output_norm))}</td>"
            f"<td>{html.escape(_fmt(layer.cross_attention_output_norm))}</td>"
            f"<td>{html.escape(_fmt(layer.mlp_activation_norm))}</td>"
            f"<td>{html.escape(_fmt(layer.mlp_output_norm))}</td></tr>"
        )
        if layer.residual_by_token is not None:
            residual_sections.append(
                f"<h3>{html.escape(layer.name)} Residual Norms</h3>"
                + _heatmap_table(
                    layer.residual_by_token,
                    tokens=tokens,
                    max_rows=max_tokens,
                    max_cols=max_tokens,
                )
            )
        if layer.mlp_activation is not None or layer.mlp_output is not None:
            residual_sections.append(
                _summary_table(
                    f"{layer.name} MLP Summaries",
                    [
                        ("activation", layer.mlp_activation),
                        ("output", layer.mlp_output),
                    ],
                )
            )

    attention_sections = "".join(
        _attention_section(attention, tokens, max_tokens=max_tokens, max_heads=max_heads)
        for attention in trace.attentions
    )
    logits_section = (
        _logit_section(trace.logits, tokens, max_tokens=max_tokens)
        if trace.logits is not None
        else ""
    )
    metadata = html.escape(json.dumps(trace.metadata, indent=2, default=str))
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Miniformer Trace</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, sans-serif; }}
body {{ margin: 24px; color: #17202a; }}
table {{ border-collapse: collapse; margin: 12px 0 24px; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #d7dde5; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #eef2f6; }}
section {{ border-top: 1px solid #d7dde5; padding-top: 16px; margin-top: 18px; }}
.notice {{ background: #fff7d6; border: 1px solid #e7cb62; padding: 8px; }}
.heatmap td {{ min-width: 44px; font-variant-numeric: tabular-nums; }}
pre {{ background: #f6f8fa; padding: 12px; overflow: auto; }}
</style>
</head>
<body>
<h1>Miniformer Trace</h1>
<p>Output shape: {html.escape(_shape_text(trace.output_shape))}</p>
<section><h2>Cache Status</h2><p>{html.escape(cache_status)}</p>
<p>{html.escape(cache.reason or "")}</p></section>
<section><h2>Layer Table</h2>
<table><tr><th>Layer</th><th>Shape</th><th>Input norm</th><th>Output norm</th>
<th>Residual delta</th><th>Self-attn norm</th><th>Cross-attn norm</th>
<th>MLP activation</th><th>MLP output</th></tr>{''.join(layer_rows)}</table>
{''.join(residual_sections)}</section>
<section><h2>Attention</h2>{attention_sections}</section>
{logits_section}
<section><h2>Metadata</h2><pre>{metadata}</pre></section>
</body>
</html>
"""
    Path(path).write_text(document)


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
    mlp_activation: Optional[TensorSummary] = None
    mlp_output: Optional[TensorSummary] = None


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


def _attention_trace(
    name: str,
    tensor: Optional[torch.Tensor],
    *,
    q_projection: Optional[TensorSummary] = None,
    k_projection: Optional[TensorSummary] = None,
    v_projection: Optional[TensorSummary] = None,
    include_raw_attention: bool,
    max_report_tokens: int,
    max_report_heads: int,
) -> AttentionTrace:
    if tensor is None:
        return AttentionTrace(
            name=name,
            shape=(),
            entropy=None,
            per_head_entropy=None,
            max_probability=None,
            mean_probability=None,
            q_projection=q_projection,
            k_projection=k_projection,
            v_projection=v_projection,
            available=False,
            reason="attention weights unavailable; use_sdpa=True does not return weights",
        )

    values = tensor.detach().float()
    clamped = values.clamp_min(1e-12)
    token_entropy = -(clamped * clamped.log()).sum(dim=-1)
    entropy = token_entropy.mean()
    per_head_entropy = token_entropy.mean(dim=(0, 2))
    weights = None
    reason: Optional[str] = (
        "raw attention weights not stored; capture with include_raw_attention=True"
    )
    if include_raw_attention:
        head_count = values.size(1)
        query_tokens = values.size(-2)
        key_tokens = values.size(-1)
        if head_count > max_report_heads:
            reason = (
                "raw attention weights omitted because head count exceeds "
                f"max_report_heads={max_report_heads}"
            )
        elif query_tokens > max_report_tokens or key_tokens > max_report_tokens:
            reason = (
                "raw attention weights omitted because token dimensions exceed "
                f"max_report_tokens={max_report_tokens}"
            )
        else:
            weights = values.mean(dim=0).cpu().tolist()
            reason = None
    return AttentionTrace(
        name=name,
        shape=tuple(values.shape),
        entropy=float(entropy.cpu()),
        per_head_entropy=[float(item) for item in per_head_entropy.cpu().tolist()],
        max_probability=float(values.max().cpu()),
        mean_probability=float(values.mean().cpu()),
        q_projection=q_projection,
        k_projection=k_projection,
        v_projection=v_projection,
        weights=weights,
        reason=reason,
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
    probabilities = values.softmax(dim=-1)
    entropy = -(probabilities.clamp_min(1e-12) * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    return LogitTrace(
        shape=tuple(values.shape),
        top_k=k,
        token_ids=top_indices.cpu().tolist(),
        values=top_values.cpu().tolist(),
        probabilities=top_probabilities.cpu().tolist(),
        entropy=entropy.cpu().tolist(),
        top_token_ids=top_indices[..., 0].cpu().tolist(),
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
    if (input_ids == 0).any():
        return CacheTrace(
            attempted=True,
            supported=False,
            atol=atol,
            rtol=rtol,
            reason="encoder cache comparison does not support padding tokens",
        )
    if model_kwargs.get("mask") is not None:
        return CacheTrace(
            attempted=True,
            supported=False,
            atol=atol,
            rtol=rtol,
            reason="custom masks are not supported by the cache comparison probe",
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
            reason="target padding is not supported by the cache comparison probe",
        )
    if any(model_kwargs.get(key) is not None for key in ("src_mask", "tgt_mask", "memory_mask")):
        return CacheTrace(
            attempted=True,
            supported=False,
            atol=atol,
            rtol=rtol,
            reason="custom masks are not supported by the cache comparison probe",
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
    memory = model_any.encoder(src, src_mask).hidden_states

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
            residual_delta = state.output_tensor - state.input_tensor
            residual_delta_norm = _safe_norm(residual_delta)
            residual_by_token = residual_delta.detach().float().norm(dim=-1).cpu().tolist()
        else:
            residual_by_token = None
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
                residual_by_token=residual_by_token,
                self_attention_output_norm=state.self_attention_output_norm,
                cross_attention_output_norm=state.cross_attention_output_norm,
                mlp_activation_norm=state.mlp_activation_norm,
                mlp_output_norm=state.mlp_output_norm,
                mlp_activation=state.mlp_activation,
                mlp_output=state.mlp_output,
            )
        )
    return layers


def capture_transformer_trace(
    model: torch.nn.Module,
    *model_args: Any,
    layer_name_fragments: Sequence[str] = (".layers.",),
    top_k: int = 5,
    include_raw_attention: bool = False,
    include_logits: bool = True,
    max_report_tokens: int = DEFAULT_MAX_REPORT_TOKENS,
    max_report_heads: int = DEFAULT_MAX_REPORT_HEADS,
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
    if not isinstance(include_raw_attention, bool):
        raise TypeError("include_raw_attention must be a boolean")
    if not isinstance(include_logits, bool):
        raise TypeError("include_logits must be a boolean")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    _validate_positive_int("max_report_tokens", max_report_tokens)
    _validate_positive_int("max_report_heads", max_report_heads)

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
        module: torch.nn.Module,
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
        q_summary = None
        k_summary = None
        v_summary = None
        if len(_inputs) >= 3 and all(isinstance(item, torch.Tensor) for item in _inputs[:3]):
            attention_module = cast(Any, module)
            q_summary = _summary(f"{name}.q_projection", attention_module.wq(_inputs[0]))
            k_summary = _summary(f"{name}.k_projection", attention_module.wk(_inputs[1]))
            v_summary = _summary(f"{name}.v_projection", attention_module.wv(_inputs[2]))
        attentions.append(
            _attention_trace(
                name,
                attn,
                q_projection=q_summary,
                k_projection=k_summary,
                v_projection=v_summary,
                include_raw_attention=include_raw_attention,
                max_report_tokens=max_report_tokens,
                max_report_heads=max_report_heads,
            )
        )

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
            state.mlp_output = _summary(f"{name}.output", tensor)
        if inputs and isinstance(inputs[0], torch.Tensor):
            activation = _feed_forward_activation(module, inputs[0])
            if activation is not None:
                state.mlp_activation_norm = _safe_norm(activation)
                state.mlp_activation = _summary(f"{name}.activation", activation)

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

    logits = (
        _logit_trace(output_tensor, top_k)
        if include_logits and _is_logit_like(model, output_tensor)
        else None
    )
    return TransformerTrace(
        output_shape=tuple(output_tensor.shape),
        layers=_build_layer_traces(layer_order, layer_states),
        attentions=attentions,
        logits=logits,
        cache=cache,
        metadata={
            "model_class": model.__class__.__name__,
            "training_restored_to": was_training,
            "include_raw_attention": include_raw_attention,
            "include_logits": include_logits,
            "max_report_tokens": max_report_tokens,
            "max_report_heads": max_report_heads,
        },
    )


def plot_trace_summary(trace: TransformerTrace):
    """Plot residual, attention, and MLP norms from a captured trace."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Install miniformer[viz] to use plotting helpers") from exc

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
