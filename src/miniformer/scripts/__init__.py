"""Small command-line helpers for repository and example workflows."""

from miniformer.scripts.make_tiny_jsonl import write_all_tiny_jsonl, write_tiny_jsonl
from miniformer.scripts.run_checks import CHECKS, run_checks
from miniformer.scripts.write_trace_report import write_trace_report

__all__ = [
    "CHECKS",
    "run_checks",
    "write_all_tiny_jsonl",
    "write_tiny_jsonl",
    "write_trace_report",
]
