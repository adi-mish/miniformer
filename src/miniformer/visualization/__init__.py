"""Visualization helpers."""

from miniformer.inspect import (
    DEFAULT_MAX_REPORT_HEADS,
    DEFAULT_MAX_REPORT_TOKENS,
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
from miniformer.visualization.visualize import plot_attention, visualize_embeddings

__all__ = [
    "AttentionTrace",
    "CacheTrace",
    "DEFAULT_MAX_REPORT_HEADS",
    "DEFAULT_MAX_REPORT_TOKENS",
    "LayerTrace",
    "LogitTrace",
    "TensorSummary",
    "TransformerTrace",
    "capture_transformer_trace",
    "plot_attention",
    "plot_trace_summary",
    "save_trace_html",
    "visualize_embeddings",
]
