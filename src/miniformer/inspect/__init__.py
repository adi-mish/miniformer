"""Structured model inspection APIs."""

from miniformer.inspect.trace import (
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
    "plot_trace_summary",
    "save_trace_html",
]
