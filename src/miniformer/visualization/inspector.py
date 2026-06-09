"""Backward-compatible imports for the structured inspection API."""

from miniformer.inspect import (
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
