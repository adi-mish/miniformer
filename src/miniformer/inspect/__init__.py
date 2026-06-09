"""Structured model inspection APIs."""

from miniformer.inspect.trace import (
    AttentionTrace,
    CacheTrace,
    LayerTrace,
    LogitTrace,
    TensorSummary,
    TransformerTrace,
    capture_transformer_trace,
    plot_trace_summary,
    save_trace_html,
)

__all__ = [
    "AttentionTrace",
    "CacheTrace",
    "LayerTrace",
    "LogitTrace",
    "TensorSummary",
    "TransformerTrace",
    "capture_transformer_trace",
    "plot_trace_summary",
    "save_trace_html",
]
